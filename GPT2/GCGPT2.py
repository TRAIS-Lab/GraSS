from transformers import GPT2PreTrainedModel, GPT2Config
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.nn.parameter import Parameter

class GCLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, transpose=False):
        super().__init__(in_features, out_features, bias)
        self.pre_activation = None
        self.layer_input = None
        self.name = 'linear'
        self.has_bias = bias
        self.transpose = transpose

        # If transpose is True, we need to handle the weight differently
        if transpose:
            self.weight = Parameter(torch.empty((in_features, out_features)))
            if bias:
                self.bias = Parameter(torch.empty(out_features))
            else:
                self.register_parameter('bias', None)
            self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.weight, mean=0.0, std=0.02)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, input):
        self.layer_input = input
        if self.transpose:
            out = F.linear(input, self.weight.t(), self.bias)
        else:
            out = F.linear(input, self.weight, self.bias)
        self.pre_activation = out
        return self.pre_activation

    def per_example_gradient(self, deriv_pre_activ):
        is_2d = self.layer_input.dim() == 2
        H = self.layer_input

        if is_2d:
            batch_size = deriv_pre_activ.size(0)
            dLdZ = deriv_pre_activ * batch_size

            pe_grad_weight = torch.bmm(dLdZ.view(batch_size, -1, 1),
                                     H.view(batch_size, 1, -1))
            if self.transpose:
                pe_grad_weight = pe_grad_weight.transpose(-1, -2)
            pe_grad_bias = dLdZ
        else:
            dLdZ = deriv_pre_activ.permute(1, 2, 0)
            dLdZ *= dLdZ.size(0)
            pe_grad_weight = torch.bmm(dLdZ,
                                     H.transpose(0, 1))
            if self.transpose:
                pe_grad_weight = pe_grad_weight.transpose(-1, -2)
            pe_grad_bias = dLdZ.sum(dim=-1)

        return pe_grad_weight, pe_grad_bias

    def pe_grad_sqnorm(self, deriv_pre_activ):
        is_2d = self.layer_input.dim() == 2
        H = self.layer_input

        if is_2d:
            batch_size = deriv_pre_activ.size(0)
            dLdZ = deriv_pre_activ * batch_size

            zsum = dLdZ.pow(2).sum(1)
            hsum = H.pow(2).sum(1)
            s = zsum * hsum

            return s + zsum
        else:
            pe_grad_weight, pe_grad_bias = self.per_example_gradient(deriv_pre_activ)
            batch_size = pe_grad_weight.size(0)
            sq_norm_weight = pe_grad_weight.pow(2).view(batch_size, -1).sum(1)
            sq_norm_bias = pe_grad_bias.pow(2).view(batch_size, -1).sum(1)

            return sq_norm_weight + sq_norm_bias

    def pe_grad_gradcomp(self, deriv_pre_activ, per_sample=True):
        is_2d = self.layer_input.dim() == 2
        H = self.layer_input
        batch_size = deriv_pre_activ.shape[0]
        dLdZ = deriv_pre_activ * batch_size

        if is_2d and self.has_bias:
            ones_column = torch.ones(H.size(0), 1, device=H.device)
            H = torch.cat((H, ones_column), dim=1)

        return dLdZ, H

class CustomGPT2Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = CustomGPT2Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = CustomGPT2MLP(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        attn_output = self.attn(
            self.ln_1(hidden_states),
            attention_mask=attention_mask,
            head_mask=head_mask
        )
        hidden_states = hidden_states + attn_output
        feed_forward_hidden_states = self.mlp(self.ln_2(hidden_states))
        hidden_states = hidden_states + feed_forward_hidden_states
        return hidden_states

class CustomGPT2Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.attn_pdrop
        # Note the transpose=True for these layers
        self.c_attn = GCLinear(config.n_embd, 3 * config.n_embd, transpose=True)
        self.c_proj = GCLinear(config.n_embd, config.n_embd, transpose=True)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((config.max_position_embeddings, config.max_position_embeddings)))
            .view(1, 1, config.max_position_embeddings, config.max_position_embeddings)
        )

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        B, T, C = hidden_states.size()
        q, k, v = self.c_attn(hidden_states).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))

class CustomGPT2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Note the transpose=True for these layers
        self.c_fc = GCLinear(config.n_embd, 4 * config.n_embd, transpose=True)
        self.c_proj = GCLinear(4 * config.n_embd, config.n_embd, transpose=True)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class CustomGPT2Model(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.max_position_embeddings, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([CustomGPT2Layer(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None):
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]

        if position_ids is None:
            position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        for layer in self.h:
            hidden_states = layer(hidden_states, attention_mask)

        hidden_states = self.ln_f(hidden_states)
        return hidden_states

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

class CustomGPT2LMHeadModel(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = CustomGPT2Model(config)
        self.lm_head = GCLinear(config.n_embd, config.vocab_size, bias=False)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None, token_type_ids=None, position_ids=None):
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids
        )

        hidden_states = transformer_outputs
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            hidden_states=hidden_states,
        )

    def get_input_embeddings(self):
        return self.transformer.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.transformer.set_input_embeddings(new_embeddings)


    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {
            "input_ids": input_ids,
            "attention_mask": kwargs.get("attention_mask", None),
            "token_type_ids": kwargs.get("token_type_ids", None),
            "position_ids": kwargs.get("position_ids", None),
        }