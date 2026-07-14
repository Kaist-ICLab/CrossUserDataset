"""CGDM: Cross-Domain Gradient Discrepancy Minimization (Du et al., CVPR 2021)."""

import copy
from itertools import cycle

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score
from torch.autograd import grad
from tqdm import tqdm

from .._da_helpers import _finalize_training_metadata


class GradientReversalLayer(autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class CGDMBackbone(nn.Module):
    """
    CGDM ResBottle (MLP for tabular features).
    """

    def __init__(self, input_dim=4):
        super(CGDMBackbone, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dim = 128

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(self.bn1(out))
        out = self.fc2(out)
        out = F.relu(self.bn2(out))
        out = self.fc3(out)
        out = F.relu(self.bn3(out))
        return out

    def output_num(self):
        return self.dim


def grad_reverse(x, lambd=1.0):
    return GradientReversalLayer.apply(x, lambd)


class CGDMClassifier(nn.Module):
    """
    CGDM ResClassifier (MLP classifier head).
    """

    def __init__(self, num_classes=3, num_layer=4, num_unit=128, prob=0.5, middle=64):
        super(CGDMClassifier, self).__init__()
        layers = []
        in_dim = num_unit

        layers.append(nn.Dropout(p=prob))
        layers.append(nn.Linear(in_dim, middle))
        layers.append(nn.BatchNorm1d(middle))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(num_layer - 1):
            layers.append(nn.Dropout(p=prob))
            layers.append(nn.Linear(middle, middle))
            layers.append(nn.BatchNorm1d(middle))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Linear(middle, num_classes))
        self.classifier = nn.Sequential(*layers)
        self.lambd = 1.0

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
        x = self.classifier(x)
        return x


class CGDM(nn.Module):
    """
    CGDM model container: feature extractor + two classifiers.
    """

    def __init__(self, input_dim=8, num_classes=6):
        super(CGDM, self).__init__()
        self.feature_extractor = CGDMBackbone(input_dim=input_dim)
        self.classifier1 = CGDMClassifier(
            num_classes=num_classes,
            num_layer=2,
            num_unit=self.feature_extractor.output_num(),
            middle=1000,
        )
        self.classifier2 = CGDMClassifier(
            num_classes=num_classes,
            num_layer=2,
            num_unit=self.feature_extractor.output_num(),
            middle=1000,
        )

    def forward(self, x):
        feat = self.feature_extractor(x)
        out1 = self.classifier1(feat)
        out2 = self.classifier2(feat)
        return out1, out2

    def predict(self, x):
        out1, out2 = self.forward(x)
        return out1 + out2


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)
    elif classname.find("Linear") != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)


def discrepancy(out1, out2):
    return torch.mean(torch.abs(F.softmax(out1, dim=1) - F.softmax(out2, dim=1)))


def Entropy_div(input_):
    epsilon = 1e-5
    input_ = torch.mean(input_, 0) + epsilon
    entropy = input_ * torch.log(input_)
    entropy = torch.sum(entropy)
    return entropy


def Entropy_condition(input_):
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1).mean()
    return entropy


def Entropy(input_):
    return Entropy_condition(input_) + Entropy_div(input_)


def Weighted_CrossEntropy(input_, labels):
    input_s = F.softmax(input_, dim=1)
    entropy = -input_s * torch.log(input_s + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    weight = 1.0 + torch.exp(-entropy)
    weight = weight / torch.sum(weight).detach().item()
    return torch.mean(weight * nn.CrossEntropyLoss(reduction="none")(input_, labels))


def obtain_label(loader, netE, netC1, netC2, device):
    start_test = True
    netE.eval()
    netC1.eval()
    netC2.eval()
    with torch.no_grad():
        for data in loader:
            inputs, labels = data[0], data[1]
            inputs = inputs.to(device)
            feas = netE(inputs)
            outputs1 = netC1(feas)
            outputs2 = netC2(feas)
            outputs = outputs1 + outputs2
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    label_np = all_label.float().cpu().numpy()
    has_labels = np.unique(label_np).size > 1
    if has_labels:
        accuracy = torch.sum(predict.to(device).float() == all_label.to(device)).item() / float(
            all_label.size()[0]
        )

    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
    all_fea = all_fea.float().cpu().numpy()

    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
    dd = cdist(all_fea, initc, "cosine")
    pred_label = dd.argmin(axis=1)
    if has_labels:
        acc = np.sum(pred_label == label_np) / len(all_fea)

    for _ in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(all_fea, initc, "cosine")
        pred_label = dd.argmin(axis=1)
        if has_labels:
            acc = np.sum(pred_label == label_np) / len(all_fea)

    return pred_label.astype("int")


def gradient_discrepancy_loss_margin(p_s1, p_s2, s_y, p_t1, p_t2, t_y, netE, netC1, netC2):
    loss_w = Weighted_CrossEntropy
    loss = nn.CrossEntropyLoss()

    src_loss1 = loss(p_s1, s_y)
    tgt_loss1 = loss_w(p_t1, t_y)

    src_loss2 = loss(p_s2, s_y)
    tgt_loss2 = loss_w(p_t2, t_y)

    grad_cossim11 = []
    grad_cossim22 = []

    for _, p in netC1.named_parameters():
        real_grad = grad([src_loss1], [p], create_graph=True, only_inputs=True, allow_unused=False)[
            0
        ]
        fake_grad = grad([tgt_loss1], [p], create_graph=True, only_inputs=True, allow_unused=False)[
            0
        ]

        if len(p.shape) > 1:
            _cossim = F.cosine_similarity(fake_grad, real_grad, dim=1).mean()
        else:
            _cossim = F.cosine_similarity(fake_grad, real_grad, dim=0)
        grad_cossim11.append(_cossim)

    grad_cossim1 = torch.stack(grad_cossim11)
    gm_loss1 = (1.0 - grad_cossim1).mean()

    for _, p in netC2.named_parameters():
        real_grad = grad([src_loss2], [p], create_graph=True, only_inputs=True)[0]
        fake_grad = grad([tgt_loss2], [p], create_graph=True, only_inputs=True)[0]

        if len(p.shape) > 1:
            _cossim = F.cosine_similarity(fake_grad, real_grad, dim=1).mean()
        else:
            _cossim = F.cosine_similarity(fake_grad, real_grad, dim=0)
        grad_cossim22.append(_cossim)

    grad_cossim2 = torch.stack(grad_cossim22)
    gm_loss2 = (1.0 - grad_cossim2).mean()
    gm_loss = (gm_loss1 + gm_loss2) / 2.0

    return gm_loss


def train_cgdm(
    model,
    X_source,
    y_source,
    X_target,
    y_target=None,
    X_val=None,
    y_val=None,
    epochs=20,
    batch_size=64,
    lr=1e-4,
    weight_decay=5e-4,
    num_k=4,
    log_interval=100,
    patience=20,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    model = model.to(device)

    if y_target is None:
        y_target = np.zeros(len(X_target), dtype=np.int64)

    X_source_t = torch.tensor(X_source, dtype=torch.float32)
    y_source_t = torch.tensor(y_source, dtype=torch.long)
    X_target_t = torch.tensor(X_target, dtype=torch.float32)
    y_target_t = torch.tensor(y_target, dtype=torch.long)

    class IndexedTensorDataset(torch.utils.data.Dataset):
        def __init__(self, x_tensor, y_tensor):
            self.x = x_tensor
            self.y = y_tensor

        def __len__(self):
            return len(self.x)

        def __getitem__(self, index):
            return self.x[index], self.y[index], index

    train_dataset = torch.utils.data.TensorDataset(X_source_t, y_source_t)
    test_dataset = IndexedTensorDataset(X_target_t, y_target_t)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    test_loader_eval = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )

    val_loader = None
    if X_val is not None and y_val is not None:
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.long)
        val_dataset = torch.utils.data.TensorDataset(X_val_t, y_val_t)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, drop_last=False
        )

    G = model.feature_extractor
    F1 = model.classifier1
    F2 = model.classifier2

    F1.apply(weights_init)
    F2.apply(weights_init)

    optimizer_g = torch.optim.Adam(G.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer_f = torch.optim.Adam(
        list(F1.parameters()) + list(F2.parameters()), lr=lr, weight_decay=weight_decay
    )

    start = 0
    criterion = nn.CrossEntropyLoss()

    def _eval_val():
        if val_loader is None:
            return None, None
        model.eval()
        all_probs, all_targets = [], []
        total_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model.predict(xb)
                total_loss += criterion(logits, yb).item()
                all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
                all_targets.append(yb.cpu().numpy())
        total_loss /= max(1, len(val_loader))
        all_probs = np.concatenate(all_probs, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        try:
            val_auroc = (
                roc_auc_score(all_targets, all_probs[:, 1])
                if all_probs.shape[1] == 2
                else roc_auc_score(all_targets, all_probs, multi_class="ovr")
            )
        except Exception:
            val_auroc = 0.5
        return total_loss, val_auroc

    best_val_score = -float("inf")
    best_model_state = None
    patience_counter = 0
    mem_label = None
    best_epoch = None
    early_stopped = False
    early_stop_epoch = None
    epochs_ran = 0
    epoch_history = []

    epoch_iterator = tqdm(range(epochs), desc="CGDM Training")
    for ep in epoch_iterator:
        iter_source = iter(train_loader)
        iter_target = cycle(test_loader)
        steps = len(train_loader)

        # Pseudo labels are required when ep > start.
        # Ensure we bootstrap once before first use, then refresh every 3 epochs.
        if ep > start and (mem_label is None or ep % 3 == 0):
            mem_label = obtain_label(test_loader_eval, G, F1, F2, device)
            mem_label = torch.from_numpy(mem_label).to(device)

        for batch_idx in range(steps - 1):
            G.train()
            F1.train()
            F2.train()

            try:
                data_s, label_s = next(iter_source)
            except StopIteration:
                iter_source = iter(train_loader)
                data_s, label_s = next(iter_source)

            data_t, label_t, idx_t = next(iter_target)

            if ep > start:
                pseudo_label_t = mem_label[idx_t].to(device)

            data_s, label_s = data_s.to(device), label_s.to(device)
            data_t, label_t = data_t.to(device), label_t.to(device)
            data_all = torch.cat((data_s, data_t), 0)
            bs = len(label_s)

            # Step A: train G, F1, F2 jointly
            optimizer_g.zero_grad()
            optimizer_f.zero_grad()
            output = G(data_all)
            output1 = F1(output)
            output2 = F2(output)
            output_s1, output_s2 = output1[:bs], output2[:bs]
            output_t1, output_t2 = output1[bs:], output2[bs:]
            output_t1_s = F.softmax(output_t1, dim=1)
            output_t2_s = F.softmax(output_t2, dim=1)

            entropy_loss = Entropy(output_t1_s) + Entropy(output_t2_s)
            supervision_loss = (
                (
                    Weighted_CrossEntropy(output_t1, pseudo_label_t)
                    + Weighted_CrossEntropy(output_t2, pseudo_label_t)
                )
                if ep > start
                else 0
            )
            loss1 = criterion(output_s1, label_s)
            loss2 = criterion(output_s2, label_s)
            all_loss = loss1 + loss2 + 0.01 * entropy_loss + 0.01 * supervision_loss
            all_loss.backward()
            optimizer_g.step()
            optimizer_f.step()

            # Step B: train F1, F2 to maximize discrepancy
            optimizer_g.zero_grad()
            optimizer_f.zero_grad()
            output = G(data_all)
            output1 = F1(output)
            output2 = F2(output)
            output_s1, output_s2 = output1[:bs], output2[:bs]
            output_t1, output_t2 = output1[bs:], output2[bs:]
            output_t1_s = F.softmax(output_t1, dim=1)
            output_t2_s = F.softmax(output_t2, dim=1)

            loss1 = criterion(output_s1, label_s)
            loss2 = criterion(output_s2, label_s)
            entropy_loss = Entropy(output_t1_s) + Entropy(output_t2_s)
            loss_dis = discrepancy(output_t1, output_t2)
            all_loss = loss1 + loss2 - 1.0 * loss_dis + 0.01 * entropy_loss
            all_loss.backward()
            optimizer_f.step()

            # Step C: train G to minimize discrepancy (num_k steps)
            for _ in range(num_k):
                optimizer_g.zero_grad()
                optimizer_f.zero_grad()
                output = G(data_all)
                output1 = F1(output)
                output2 = F2(output)
                output_s1, output_s2 = output1[:bs], output2[:bs]
                output_t1, output_t2 = output1[bs:], output2[bs:]
                output_t1_s = F.softmax(output_t1, dim=1)
                output_t2_s = F.softmax(output_t2, dim=1)

                entropy_loss = Entropy(output_t1_s) + Entropy(output_t2_s)
                loss_dis = discrepancy(output_t1, output_t2)
                gmn_loss = (
                    gradient_discrepancy_loss_margin(
                        output_s1,
                        output_s2,
                        label_s,
                        output_t1,
                        output_t2,
                        pseudo_label_t,
                        G,
                        F1,
                        F2,
                    )
                    if ep > start
                    else 0
                )

                all_loss = 1.0 * loss_dis + 0.01 * entropy_loss + 0.01 * gmn_loss
                all_loss.backward()
                optimizer_g.step()

        val_loss, val_auroc = _eval_val()
        if val_loss is not None:
            epoch_iterator.set_postfix(
                {"Val Loss": f"{val_loss:.4f}", "Val AUC": f"{val_auroc:.4f}"}
            )
            epoch_num = ep + 1
            epochs_ran = epoch_num
            epoch_history.append(
                {
                    "epoch": epoch_num,
                    "val_loss": round(float(val_loss), 6),
                    "val_auroc": round(float(val_auroc), 6),
                }
            )
            if val_auroc > best_val_score:
                best_val_score = val_auroc
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
                best_epoch = epoch_num
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    epoch_iterator.write(f"Early stopping at epoch {ep}")
                    early_stopped = True
                    early_stop_epoch = epoch_num
                    break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    _finalize_training_metadata(
        model,
        optimizer="Adam",
        best_epoch=best_epoch,
        early_stopped=early_stopped,
        early_stop_epoch=early_stop_epoch,
        epochs_ran=epochs_ran if epochs_ran else epochs,
        max_epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        lr=lr,
        weight_decay=weight_decay,
        best_metric_value=round(float(best_val_score), 6) if best_epoch is not None else None,
        epoch_history=epoch_history,
        extra={"num_k": num_k},
    )
    return model
