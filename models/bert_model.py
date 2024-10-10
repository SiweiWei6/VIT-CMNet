import torch
import os
from torch import nn
import torch.nn.functional as F
from torchcrf import CRF
from .modeling_bert import BertModel
from transformers.modeling_outputs import TokenClassifierOutput
from torchvision.models import resnet50
from transformers.models.clip import CLIPVisionModel



class ImageModel(nn.Module):
    def __init__(self,clip_vision_model):
        super(ImageModel, self).__init__()

        # self.resnet = resnet50(pretrained=True)
        self.clip_vision_model=clip_vision_model.to('cuda')
    
    def forward(self, clip_pixel_values, aux_clip_pixel_values):
        # full image prompt
        prompt_guids = self.clip_vision_model.get_FPN_vision_feature(pixel_values=clip_pixel_values)

        # aux_imgs: bsz x 3(nums) x 3 x 224 x 224
        if aux_clip_pixel_values is not None:
            aux_prompt_guids = []
            aux_clip_pixel_values = aux_clip_pixel_values.permute(
                [1, 0, 2, 3, 4])
            for i in range(len(aux_clip_pixel_values)):
                aux_prompt_guid = self.clip_vision_model.get_FPN_vision_feature(
                    pixel_values=aux_clip_pixel_values[i])
                aux_prompt_guids.append(aux_prompt_guid)
            return prompt_guids, aux_prompt_guids
        return prompt_guids, None

    def get_resnet_prompt(self, x):
        """generate image prompt

        Args:
            x ([torch.tenspr]): bsz x 3 x 224 x 224

        Returns:
            prompt_guids ([List[torch.tensor]]): 4 x List[bsz x 256 x 7 x 7]
        """
        # image: bsz x 3 x 224 x 224
        prompt_guids = []
        for name, layer in self.resnet.named_children():
            if name == 'fc' or name == 'avgpool':  continue
            x = layer(x)
            if 'layer' in name:
                bsz, channel, ft, _ = x.size()
                kernel = ft // 2
                prompt_kv = nn.AvgPool2d(kernel_size=(kernel, kernel), stride=kernel)(x)    # (bsz, 256, 7, 7)
                prompt_guids.append(prompt_kv)
        return prompt_guids


class HMNeTREModel(nn.Module):
    def __init__(self, num_labels, tokenizer,clip_preprocessor, args):
        super(HMNeTREModel, self).__init__()
        self.bert = BertModel.from_pretrained(args.bert_name)
        self.clip_vision_model =CLIPVisionModel.from_pretrained(args.clip_name)
        self.bert.resize_token_embeddings(len(tokenizer))
        self.args = args

        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(self.bert.config.hidden_size*2, num_labels)
        self.head_start = tokenizer.convert_tokens_to_ids("<s>")
        self.tail_start = tokenizer.convert_tokens_to_ids("<o>")
        self.tokenizer = tokenizer
        self.clip_preprocessor=clip_preprocessor#clip


        if self.args.use_prompt:
            self.image_model = ImageModel(self.clip_vision_model)

            self.encoder_conv =  nn.Sequential(
                                    nn.Linear(in_features=12*768, out_features=800),
                                    nn.Tanh(),
                                    nn.Linear(in_features=800, out_features=12*2*768)
                                )

            self.gates = nn.ModuleList([nn.Linear(1 * 768 * 1, 12) for i in range(12)])
            self.encoder_text = nn.Linear(768, 2 * 768)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        images=None,
        aux_imgs=None,
        # caption input
        seq_input_ids=None,
        seq_token_type_ids=None,
        seq_attention_mask=None,
        # check result
        seq=None,
        img=None,
        relation=None,
    ):
        # caption
        seq_output = self.bert(
            input_ids=seq_input_ids,
            token_type_ids=seq_token_type_ids,
            attention_mask=seq_attention_mask,
            output_attentions=True,
            return_dict=True
        )
        seq_last_hidden_state, _ = seq_output.last_hidden_state, seq_output.pooler_output
        bsz = input_ids.size(0)
        if self.args.use_prompt:
            clip_pixel_values=[]
            for image in images:
                clip_result = self.clip_preprocessor(images=image.to('cpu'), padding='max_length',
                                                max_length=self.args.image_max_length, truncation=True)
                clip_pixel_values.append(clip_result['pixel_values'][0])
            aux_clip_pixel_values = []
            if aux_imgs is not None:
                for aux_img in aux_imgs:
                    aux_clip_pixel_value=[]
                    for this_aux_img in aux_img:
                        aux_clip_result = self.clip_preprocessor(images=this_aux_img.to('cpu'), padding='max_length',
                                                            max_length=self.args.image_max_length, truncation=True)
                        aux_clip_pixel_value.append(aux_clip_result['pixel_values'][0])
                    aux_clip_pixel_values.append(aux_clip_pixel_value)

            clip_pixel_values = torch.tensor(clip_pixel_values, dtype=torch.float32).to(self.args.device)
            aux_clip_pixel_values = torch.tensor(aux_clip_pixel_values, dtype=torch.float32).to(self.args.device)

            prompt_guids = self.get_visual_prompt(clip_pixel_values, aux_clip_pixel_values,self.clip_vision_model,seq_last_hidden_state)
            prompt_guids_length = prompt_guids[0][0].shape[2]
            prompt_guids_mask = torch.ones((bsz, prompt_guids_length)).to(self.args.device)
            prompt_attention_mask = torch.cat((prompt_guids_mask, attention_mask), dim=1)
        else:
            prompt_guids = None
            prompt_attention_mask = attention_mask

        output = self.bert(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=prompt_attention_mask,
                    past_key_values=prompt_guids,
                    output_attentions=True,
                    return_dict=True
        )

        last_hidden_state, pooler_output = output.last_hidden_state, output.pooler_output
        bsz, seq_len, hidden_size = last_hidden_state.shape
        entity_hidden_state = torch.Tensor(bsz, 2*hidden_size) # batch, 2*hidden
        for i in range(bsz):
            head_idx = input_ids[i].eq(self.head_start).nonzero().item()

            tail_idx = input_ids[i].eq(self.tail_start).nonzero().item()
            head_hidden = last_hidden_state[i, head_idx, :].squeeze()
            tail_hidden = last_hidden_state[i, tail_idx, :].squeeze()
            entity_hidden_state[i] = torch.cat([head_hidden, tail_hidden], dim=-1)
        entity_hidden_state = entity_hidden_state.to(self.args.device)
        logits = self.classifier(entity_hidden_state)
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            return loss_fn(logits, labels.view(-1)), logits
        return logits

    def get_visual_prompt(self, clip_pixel_values, aux_clip_pixel_values,clip_vision_model,seq_last_hidden_state):
        bsz = clip_pixel_values.size(0)
        # full image prompt
        prompt_guids, aux_prompt_guids = self.image_model(clip_pixel_values, aux_clip_pixel_values)
        prompt_guids = torch.cat(prompt_guids, dim=1).view(bsz, 1, -1)


        aux_prompt_guids = [torch.cat(aux_prompt_guid, dim=1).view(bsz, 1, -1) for aux_prompt_guid in aux_prompt_guids]
        split_prompt_guids = prompt_guids.split(768*1, dim=-1)
        split_aux_prompt_guids = [aux_prompt_guid.split(768*1, dim=-1) for aux_prompt_guid in aux_prompt_guids]
        sum_prompt_guids = torch.stack(split_prompt_guids).sum(0).view(bsz, -1)/12

        result = []
        text_key_val = F.leaky_relu(seq_last_hidden_state)##image caption#
        for idx in range(12):
            prompt_gate = F.softmax(F.leaky_relu(self.gates[idx](sum_prompt_guids)), dim=-1)

            key_val = torch.zeros_like(split_prompt_guids[0]).to(self.args.device)
            for i in range(12):
                key_val = key_val + torch.einsum('bg,blh->blh', prompt_gate[:, i].view(-1, 1), split_prompt_guids[i])

            # use gate mix aux image prompts
            aux_key_vals = []   # 3 x [bsz, 4, 768*2]
            for split_aux_prompt_guid in split_aux_prompt_guids:
                sum_aux_prompt_guids = torch.stack(split_aux_prompt_guid).sum(0).view(bsz, -1)
                aux_prompt_gate = F.softmax(F.leaky_relu(self.gates[idx](sum_aux_prompt_guids)), dim=-1)
                aux_key_val = torch.zeros_like(split_aux_prompt_guid[0]).to(self.args.device)
                for i in range(12):#4
                    aux_key_val = aux_key_val + torch.einsum('bg,blh->blh', aux_prompt_gate[:, i].view(-1, 1), split_aux_prompt_guid[i])
                aux_key_vals.append(aux_key_val)
            key_val = [key_val] + aux_key_vals + [text_key_val]
            key_val = torch.cat(key_val, dim=1)
            key_val = key_val.split(768, dim=-1)
            key, value = key_val[0].reshape(bsz, 12, -1, 64).contiguous(), key_val[0].reshape(bsz, 12, -1,64).contiguous()
            temp_dict = (key, value)
            result.append(temp_dict)
        return result


class HMNeTNERModel(nn.Module):
    def __init__(self, label_list,clip_preprocessor, args):
        super(HMNeTNERModel, self).__init__()
        self.args = args
        self.prompt_dim = args.prompt_dim
        self.prompt_len = args.prompt_len
        self.bert = BertModel.from_pretrained(args.bert_name).to('cuda')
        self.clip_vision_model = CLIPVisionModel.from_pretrained(args.clip_name)  # clip
        self.bert_config = self.bert.config

        self.clip_preprocessor = clip_preprocessor  # clip

        if args.use_prompt:
            self.image_model = ImageModel(self.clip_vision_model)
            self.encoder_conv =  nn.Sequential(
                            nn.Linear(in_features=12*768, out_features=800),
                            nn.Tanh(),
                            nn.Linear(in_features=800, out_features=12*2*768)
                            )
            self.gates = nn.ModuleList([nn.Linear(1*768*1, 12) for i in range(12)]).to('cuda')
            self.encoder_text = nn.Linear(768, 2 * 768)

        self.num_labels  = len(label_list)  # pad
        print(self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True).to('cuda')
        self.fc = nn.Linear(self.bert.config.hidden_size, self.num_labels).to('cuda')
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, images=None, aux_imgs=None,# caption input
        seq_input_ids=None,
        seq_token_type_ids=None,
        seq_attention_mask=None,
        seq=None,
        img=None,
        relation=None):
        # caption
        seq_output = self.bert(
            input_ids=seq_input_ids,
            token_type_ids=seq_token_type_ids,
            attention_mask=seq_attention_mask,
            output_attentions=True,  ##
            return_dict=True  ##
        )
        seq_last_hidden_state, _ = seq_output.last_hidden_state, seq_output.pooler_output
        if self.args.use_prompt:
            # clip_process
            clip_pixel_values = []
            for image in images:
                clip_result = self.clip_preprocessor(images=image.to('cpu'), padding='max_length',
                                                     max_length=self.args.image_max_length, truncation=True)
                clip_pixel_values.append(clip_result['pixel_values'][0])
            aux_clip_pixel_values = []
            if aux_imgs is not None:
                for aux_img in aux_imgs:
                    aux_clip_pixel_value = []
                    for this_aux_img in aux_img:
                        aux_clip_result = self.clip_preprocessor(images=this_aux_img.to('cpu'), padding='max_length',
                                                                 max_length=self.args.image_max_length, truncation=True)
                        aux_clip_pixel_value.append(aux_clip_result['pixel_values'][0])
                    aux_clip_pixel_values.append(aux_clip_pixel_value)

            clip_pixel_values = torch.tensor(clip_pixel_values, dtype=torch.float32).to(self.args.device)
            aux_clip_pixel_values = torch.tensor(aux_clip_pixel_values, dtype=torch.float32).to(self.args.device)

            prompt_guids = self.get_visual_prompt(clip_pixel_values, aux_clip_pixel_values,self.clip_vision_model,seq_last_hidden_state)
            prompt_guids_length = prompt_guids[0][0].shape[2]
            # attention_mask: bsz, seq_len
            # prompt attention， attention mask
            bsz = attention_mask.size(0)
            prompt_guids_mask = torch.ones((bsz, prompt_guids_length)).to(self.args.device)
            prompt_attention_mask = torch.cat((prompt_guids_mask, attention_mask), dim=1)
        else:
            prompt_attention_mask = attention_mask
            prompt_guids = None

        bert_output = self.bert(input_ids=input_ids,
                            attention_mask=prompt_attention_mask,
                            token_type_ids=token_type_ids,
                            past_key_values=prompt_guids,
                            return_dict=True)
        sequence_output = bert_output['last_hidden_state']  # bsz, len, hidden
        sequence_output = self.dropout(sequence_output)  # bsz, len, hidden
        emissions = self.fc(sequence_output)    # bsz, len, labels
        
        logits = self.crf.decode(emissions.to('cuda'), attention_mask.byte().to('cuda'))
        loss = None
        if labels is not None:
            loss = -1 * self.crf(emissions, labels, mask=attention_mask.byte(), reduction='mean') 
        return TokenClassifierOutput(
            loss=loss,
            logits=logits
        )

    def get_visual_prompt(self, clip_pixel_values, aux_clip_pixel_values,clip_vision_model,seq_last_hidden_state):
        bsz = clip_pixel_values.size(0)
        # full image prompt
        prompt_guids, aux_prompt_guids = self.image_model(clip_pixel_values, aux_clip_pixel_values)
        prompt_guids = torch.cat(prompt_guids, dim=1).view(bsz, 1, -1)
        aux_prompt_guids = [torch.cat(aux_prompt_guid, dim=1).view(bsz, 1, -1) for aux_prompt_guid in aux_prompt_guids]
        split_prompt_guids = prompt_guids.split(768*1, dim=-1)
        split_aux_prompt_guids = [aux_prompt_guid.split(768*1, dim=-1) for aux_prompt_guid in aux_prompt_guids]

        result = []
        text_key_val = F.leaky_relu(seq_last_hidden_state)  ##image caption#
        for idx in range(12):  # 12
            sum_prompt_guids = torch.stack(split_prompt_guids).sum(0).view(bsz, -1) / 12
            prompt_gate = F.softmax(F.leaky_relu(self.gates[idx](sum_prompt_guids)), dim=-1)

            key_val = torch.zeros_like(split_prompt_guids[0]).to(self.args.device)
            for i in range(12):
                key_val = key_val + torch.einsum('bg,blh->blh', prompt_gate[:, i].view(-1, 1), split_prompt_guids[i])

            aux_key_vals = []
            for split_aux_prompt_guid in split_aux_prompt_guids:
                sum_aux_prompt_guids = torch.stack(split_aux_prompt_guid).sum(0).view(bsz, -1)
                aux_prompt_gate = F.softmax(F.leaky_relu(self.gates[idx](sum_aux_prompt_guids)), dim=-1)
                aux_key_val = torch.zeros_like(split_aux_prompt_guid[0]).to(self.args.device)
                for i in range(12):
                    aux_key_val = aux_key_val + torch.einsum('bg,blh->blh', aux_prompt_gate[:, i].view(-1, 1), split_aux_prompt_guid[i])
                aux_key_vals.append(aux_key_val)
            key_val = [key_val] + aux_key_vals
            key_val = torch.cat(key_val, dim=1)
            key_val = [key_val] + [text_key_val]  ##
            key_val = torch.cat(key_val, dim=1)
            key_val = key_val.split(768, dim=-1)
            key, value = key_val[0].reshape(bsz, 12, -1, 64).contiguous(), key_val[0].reshape(bsz, 12, -1,64).contiguous()  # bsz, 12, 4, 64
            temp_dict = (key, value)
            result.append(temp_dict)
        return result
