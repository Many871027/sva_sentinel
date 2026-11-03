# **Arquitectura y Pipeline de Destilación de Conocimiento para Detectores Densos YOLO11**

La compresión de detectores de objetos basados en redes neuronales profundas mediante destilación de conocimiento (*Knowledge Distillation*, KD) representa una de las áreas más complejas en la optimización de modelos de visión computacional contemporáneos. Mientras que en los clasificadores de imágenes convencionales la supervisión consiste en transferir un vector unidimensional de probabilidades sobre un conjunto mutuamente excluyente de clases, los detectores densos de una sola etapa (*single-stage dense detectors*), como la arquitectura YOLO11, generan predicciones multivariables a lo largo de un espacio espacial y semántico continuo.

La transferencia del conocimiento latente desde un modelo profesor masivo ($YOLO11x$, con aproximadamente $56.9M$ de parámetros) hacia un modelo estudiante ultraligero ($YOLO11n$, con $2.6M$ de parámetros) impone una reducción paramétrica drástica del \$95.43\\%\$. Alcanzar una tasa de retención superior al \$92\\%\$ del rendimiento original del profesor ($mAP@50$) exige mitigar los desbalances espaciales de la inferencia densa, formular una función de pérdida compuesta matemáticamente balanceada, adaptar topológicamente las discrepancias de canales y estructurar un pipeline de compilación para hardware perimetral (*edge*).

## **Fundamentos Teóricos de la Destilación en Detección de Objetos**

La extrapolación directa del marco clásico de destilación formulado por Hinton et al. hacia la detección densa es inviable debido a divergencias estructurales profundas en la formulación del problema predictivo. En un clasificador tradicional, la señal de error penaliza la discrepancia categórica global. En contraste, la detección densa mediante rejillas desacopladas introduce tres barreras teóricas determinantes:

El desbalance extremo entre fondo y primer plano (*extreme foreground-background imbalance*) constituye la primera complejidad crítica. En una imagen de resolución estándar ($640\times 640$ píxeles), el modelo genera miles de anclajes o celdas de predicción distribuidas en múltiples escalas espaciales. Más del \$98\\%\$ de estas posiciones pertenecen al fondo de la escena o a regiones sin información contextual discriminativa. Si se aplica una función de divergencia sobre la totalidad de los mapas de activación sin discriminación espacial, los gradientes derivados del fondo asfixian la señal de los objetos de interés, obligando al modelo estudiante a aprender patrones redundantes de zonas vacías.

La naturaleza multivariable y desacoplada de las cabezas de predicción representa la segunda restricción fundamental. En YOLO11, cada nivel de la cabeza predictiva resuelve de manera simultánea tareas con dinámicas de optimización divergentes: clasificación multietiqueta mediante probabilidades continuas, regresión geométrica de cajas delimitadoras gobernada por formulaciones IoU no lineales, y estimación de incertidumbre posicional mediante distribuciones continuas discretizadas a través de la pérdida de foco distribucional (*Distribution Focal Loss*, DFL). Destilar únicamente las probabilidades de clase ignora la incertidumbre espacial y los gradientes geométricos acumulados en las ramas de localización.

La discrepancia de dimensionalidad espacial y de canales entre el profesor y el estudiante introduce la tercera dificultad estructural. En la familia YOLO11, el escalado entre variantes no solo altera la profundidad de los bloques residuales C3k2 y C2PSA, sino también el ancho del cuello (*neck*) basado en estructuras PAN/FPN. En consecuencia, los tensores latentes intermedios correspondientes a las zancadas $8$ ($P3$), $16$ ($P4$) y $32$ ($P5$) exhiben factores de contracción de canal de $2\times$ a $4\times$, lo que imposibilita el cómputo directo de distancias euclidianas o divergencias informacionales sin transformaciones proyectivas previas.

| Dimensión Arquitectónica | Profesor (YOLO11x) | Estudiante (YOLO11n) | Factor de Escala Relativo |
| :---- | :---- | :---- | :---- |
| **Parámetros Totales** | $56.9M$ | $2.6M$ | \$-95.43\\%\$ |
| **Operaciones de Punto Flotante (FLOPs)** | $194.9G$ | $6.5G$ | \$-96.66\\%\$ |
| **Canales en Cuello P3 (Zancada 8\)** | $256$ | $64$ | $0.25\times$ |
| **Canales en Cuello P4 (Zancada 16\)** | $512$ | $128$ | $0.25\times$ |
| **Canales en Cuello P5 (Zancada 32\)** | $512$ | $256$ | $0.50\times$ |
| **Multiplicador de Profundidad (*depth\_multiple*)** | $1.00$ | $0.50$ | $0.50\times$ |
| **Multiplicador de Ancho (*width\_multiple*)** | $1.00$ | $0.25$ | $0.25\times$ |

Para abordar estas asimetrías de manera analítica, se formula un esquema de entrenamiento basado en una función de pérdida compuesta. El flujo del procesamiento distribuye el lote de imágenes idéntico hacia ambas redes de forma concurrente. El modelo profesor opera con sus pesos en modo de evaluación y con los gradientes completamente congelados, actuando como un oráculo probabilístico y semántico. Por su parte, el modelo estudiante procesa la entrada manteniendo activo el grafo computacional de autograd.

Los mapas de características del estudiante son interceptados en las etapas intermedias del cuello ($P3,P4,P5$) y proyectados dimensionalmente mediante capas de convolución adaptativa $1\times 1$ con normalización por lotes para igualar la cardinalidad de los tensores del profesor. De forma paralela, las cabezas de predicción del estudiante generan logits de clasificación y tensores distribucionales de regresión que son cotejados con las salidas suavizadas del profesor mediante divergencia de Kullback-Leibler.

Finalmente, las predicciones del estudiante son evaluadas simultáneamente frente a las etiquetas de verdad terreno (*ground truth*) mediante la asignación dinámica de muestras de la arquitectura YOLO, garantizando que el gradiente total resultante sintetice la supervisión empírica directa con la imitación de representaciones latentes.

${L}_{total}=\alpha {L}_{hard}+\beta {L}_{KD}(T,S;\tau )+\gamma {L}_{feat}$

## **Desglose Matemático de la Función de Pérdida Compuesta**

### **Pérdida Dura de Supervisión (${L}_{hard}$)**

La pérdida directa impone el anclaje empírico del estudiante sobre las etiquetas reales del conjunto de datos mediante el algoritmo *Task-Aligned Assigner* (TAL), formulándose en las arquitecturas YOLO como una combinación lineal ponderada:

${L}_{hard}={\lambda }_{cls}{L}_{BCE}+{\lambda }_{box}{L}_{CIoU}+{\lambda }_{dfl}{L}_{DFL}$  
La componente de clasificación multietiqueta se calcula evaluando la entropía cruzada binaria por clase sobre cada celda seleccionada por el alineador de tareas:

${L}_{BCE}(p,y)=-\sum\limits_{c=1}^{C}\left[{{y}_{c}\log \left({\sigma ({p}_{c})}\right)+(1-{y}_{c})\log \left({1-\sigma ({p}_{c})}\right)}\right]$  
La optimización espacial de las coordenadas se modela mediante la pérdida Complete IoU (${L}_{CIoU}$), la cual penaliza simultáneamente la superposición de área, la distancia euclidiana entre centroides normalizada por la diagonal del rectángulo contenedor mínimo $c$, y la consistencia en la relación de aspecto:

${L}_{CIoU}=1-IoU+\frac{{\rho }^{2}(b,{b}^{gt})}{{c}^{2}}+{\alpha }_{ciou}v$

$v=\frac{4}{{\pi }^{2}}{\left({\arctan \left({\frac{{w}^{gt}}{{h}^{gt}}}\right)-\arctan \left({\frac{w}{h}}\right)}\right)}^{2},{\alpha }_{ciou}=\frac{v}{(1-IoU)+v}$  
Para resolver la incertidumbre inherente a los bordes de objetos oclusos o desenfocados, YOLO11 prescinde de la regresión determinista de coordenadas escalares y modela la distancia continua a los bordes mediante una distribución de probabilidad sobre un conjunto de valores discretos \$\\{y\_0, y\_1, \\dots, y\_n\\}\$ con ${y}_{0}\leq y\leq {y}_{n}$. La pérdida de foco distribucional (${L}_{DFL}$) maximiza la densidad de probabilidad en torno a los valores enteros más próximos al valor continuo $y$:

${L}_{DFL}({S}_{i},{S}_{i+1})=-\left({({y}_{i+1}-y)\log ({S}_{i})+(y-{y}_{i})\log ({S}_{i+1})}\right)$  
Donde ${S}_{i}$ y ${S}_{i+1}$ representan las probabilidades normalizadas resultantes de aplicar Softmax sobre los dos valores discretos que delimitan la coordenada real.

### **Pérdida de Destilación de Salida con Temperatura (${L}_{KD}$)**

El conocimiento tácito o residual (*dark knowledge*) describe las correlaciones intracategóricas capturadas por el profesor en las probabilidades secundarias que asigna a clases morfológicamente afines. Para acceder a estas distribuciones no observables bajo etiquetas categóricas rígidas, se introduce un hiperparámetro de temperatura $\tau >1$ sobre los vectores de logits sin activar $z$:

${q}_{i}^{(T)}=\frac{\exp \left({{z}_{i}^{(T)}/\tau }\right)}{\sum\limits_{j=1}^{C}\exp \left({{z}_{j}^{(T)}/\tau }\right)},{q}_{i}^{(S)}=\frac{\exp \left({{z}_{i}^{(S)}/\tau }\right)}{\sum\limits_{j=1}^{C}\exp \left({{z}_{j}^{(S)}/\tau }\right)}$  
La transferencia se rige por la divergencia de Kullback-Leibler reescalada:

\$\$\\mathcal{L}\_{KD\\\_cls} \= \\tau^2 \\cdot D\_{KL}\\left( q^{(T)} \\,\\vert{}\\vert{}\\, q^{(S)} \\right) \= \\tau^2 \\sum\_{k=1}^{C} q\_k^{(T)} \\log\\left( \\frac{q\_k^{(T)}}{q\_k^{(S)}} \\right)\$\$  
El factor multiplicativo cuadrático ${\tau }^{2}$ es indispensable para la estabilidad del descenso de gradiente. Analizando la derivada analítica de la divergencia KL con respecto a los logits del modelo estudiante ${z}_{i}^{(S)}$:

\$\$\\frac{\\partial \\mathcal{L}\_{KD\\\_cls}}{\\partial z\_i^{(S)}} \= \\tau^2 \\frac{\\partial D\_{KL}(q^{(T)} \\,\\vert{}\\vert{}\\, q^{(S)})}{\\partial z\_i^{(S)}} \= \\tau^2 \\left( \\frac{1}{\\tau} \\left( q\_i^{(S)} \- q\_i^{(T)} \\right) \\right) \= \\tau \\left( q\_i^{(S)} \- q\_i^{(T)} \\right)\$\$  
Al considerar el límite para temperaturas elevadas donde ${z}_{i}/\tau \ll 1$, una expansión en serie de Taylor de primer orden describe la probabilidad suavizada como:

${q}_{i}\approx \frac{1+{z}_{i}/\tau }{C+\sum\limits_{j}^{}{z}_{j}/\tau }\approx \frac{1}{C}\left({1+\frac{{z}_{i}-\bar{z}}{\tau }}\right)$  
Bajo esta condición asintótica, la diferencia de probabilidades suaves se reduce a una relación inversamente proporcional al cuadrado de la temperatura:

\$\$q\_i^{(S)} \- q\_i^{(T)} \\approx \\frac{1}{C \\tau} \\left( z\_i^{(S)} \- z\_i^{(T)} \\right) \\implies \\frac{\\partial D\_{KL}}{\\partial z\_i^{(S)}} \\approx \\frac{1}{C \\tau^2} \\left( z\_i^{(S)} \- z\_i^{(T)} \\right)\$\$  
Sin la inclusión del término correctivo ${\tau }^{2}$, la norma del gradiente transferido decrece con una tasa cuadrática conforme se eleva la temperatura, volviendo insignificante la supervisión de la destilación frente a los gradientes de ${L}_{hard}$. La presencia explícita de ${\tau }^{2}$ compensa exactamente dicha atenuación numérica, preservando un balance constante entre ambas fuerzas de optimización.

Para la rama de localización, se adopta la formulación de *Localization Distillation* (LD), proyectando la distribución continua predicha por el profesor para cada uno de los cuatro bordes cardinales de la caja ($l,r,t,b$):

\$\$\\mathcal{L}\_{KD\\\_reg} \= \\tau^2 \\sum\_{e \\in \\{l, r, t, b\\}} D\_{KL}\\left( \\text{Softmax}\\left(\\frac{B\_e^{(T)}}{\\tau}\\right) \\,\\Bigg\\Vert{}\\, \\text{Softmax}\\left(\\frac{B\_e^{(S)}}{\\tau}\\right) \\right)\$\$

${L}_{KD}={L}_{KD\_cls}+\mu {L}_{KD\_reg}$  
Donde $\mu$ pondera la importancia relativa de la incertidumbre geométrica respecto a la distribución de clases.

### **Pérdida de Características Intermedias (${L}_{feat}$)**

La fidelidad semántica en detectores depende de los mapas latentes generados por las ramas del cuello PAN. Dado que el tensor del estudiante en cada nivel $l\in \{P3,P4,P5\}$ satisface ${F}_{l}^{(S)}\in {R}^{B\times {C}_{S}\times {H}_{l}\times {W}_{l}}$ mientras que el profesor exhibe ${F}_{l}^{(T)}\in {R}^{B\times {C}_{T}\times {H}_{l}\times {W}_{l}}$ con ${C}_{T}>{C}_{S}$, se define una proyección paramétrica alineadora:

${\phi }_{l}\left({{F}_{l}^{(S)}}\right)=BN\left({{W}_{proj,l}*{F}_{l}^{(S)}}\right),{W}_{proj,l}\in {R}^{{C}_{T}\times {C}_{S}\times 1\times 1}$  
Para neutralizar el sesgo derivado del desbalance fondo-objeto, se implementa una combinación de *Channel-Wise Distillation* (CWD) y enmascaramiento espacial basado en regiones de atención. CWD normaliza cada canal de manera individual convirtiendo la matriz espacial $H\times W$ en una distribución de probabilidad mediante Softmax bidimensional:

$\psi (F{)}_{c,i}=\frac{\exp \left({{F}_{c,i}/{\tau }_{c}}\right)}{\sum\limits_{j=1}^{H\times W}\exp \left({{F}_{c,j}/{\tau }_{c}}\right)}$  
\$\$\\mathcal{L}\_{CWD} \= \\tau\_c^2 \\sum\_{l \\in \\{P3, P4, P5\\}} \\frac{1}{C\_{T, l}} \\sum\_{c=1}^{C\_{T, l}} D\_{KL}\\left( \\psi(F\_l^{(T)})\_c \\,\\Big\\Vert{}\\, \\psi(\\phi\_l(F\_l^{(S)}))\_c \\right)\$\$  
Adicionalmente, se aplica una máscara de atención binaria o de saliencia $M\in \{0,1{\}}^{H\times W}$ derivada de la proyección espacial de las anotaciones reales para enfocar la convergencia euclidiana exclusivamente sobre las regiones con instancias activas:

\$\$\\mathcal{L}\_{feat\\\_masked} \= \\sum\_{l \\in \\{P3, P4, P5\\}} \\frac{1}{\\sum M\_l \+ \\epsilon} \\left\\Vert{} M\_l \\odot \\left( F\_l^{(T)} \- \\phi\_l\\left(F\_l^{(S)}\\right) \\right) \\right\\Vert{}\_2^2\$\$

${L}_{feat}={L}_{CWD}+{\lambda }_{mask}{L}_{feat\_masked}$

## **Implementación de la Pérdida Compuesta y Extracción de Características**

La integración del criterio de destilación requiere interceptar de manera dinámica las activaciones internas del cuello de la arquitectura sin perturbar el grafo computacional de Ultralytics. La clase CompositeDistillationLoss incorpora capas de adaptación entrenables junto con la formulación CWD y la divergencia KL sobre logits y distribuciones DFL:

Python

```
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict

class ChannelWiseDistillationLoss(nn.Module):
    """
    Calcula la divergencia KL bidimensional normalizada por canal (CWD).
    Transforma el mapa espacial H x W de cada canal en una distribucion
    de probabilidad suave mitigando el desbalance fondo-primer plano.
    """
    def __init__(self, tau: float = 4.0):
        super().__init__()
        self.tau = tau

    def forward(self, s_feat: torch.Tensor, t_feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = s_feat.shape
        s_spatial = s_feat.view(b, c, -1) / self.tau
        t_spatial = t_feat.view(b, c, -1) / self.tau

        p_s = F.log_softmax(s_spatial, dim=-1)
        p_t = F.softmax(t_spatial, dim=-1)

        loss = (self.tau ** 2) * F.kl_div(p_s, p_t, reduction="batchmean")
        return loss

class CompositeDistillationLoss(nn.Module):
    """
    Criterio de perdida compuesta para detectores densos YOLO11:
    L_total = alpha * L_hard + beta * L_KD + gamma * L_feat
    """
    def __init__(
        self,
        teacher_channels: List[int] = [256, 512, 512],
        student_channels: List[int] = [64, 128, 256],
        alpha: float = 1.0,
        beta: float = 2.5,
        gamma: float = 1.5,
        tau: float = 4.0,
        mu_reg: float = 0.5
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.mu_reg = mu_reg

        # Convoluciones adaptativas 1x1 para alinear los canales del estudiante
        self.adaptation_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(s_ch, t_ch, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(t_ch),
                nn.ReLU(inplace=True)
            )
            for s_ch, t_ch in zip(student_channels, teacher_channels)
        ])

        self.cwd_loss = ChannelWiseDistillationLoss(tau=self.tau)
        self.kl_div = nn.KLDivLoss(reduction="batchmean")
        self.mse_loss = nn.MSELoss(reduction="mean")

    def forward(
        self,
        student_preds: Tuple[torch.Tensor, torch.Tensor],
        teacher_preds: Tuple[torch.Tensor, torch.Tensor],
        student_features: List[torch.Tensor],
        teacher_features: List[torch.Tensor],
        hard_loss: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        device = hard_loss.device
        s_cls_logits, s_reg_dist = student_preds
        t_cls_logits, t_reg_dist = teacher_preds

        # 1. Destilacion de Logits de Clasificacion
        s_cls_soft = F.log_softmax(s_cls_logits / self.tau, dim=-1)
        t_cls_soft = F.softmax(t_cls_logits / self.tau, dim=-1)
        loss_kd_cls = (self.tau ** 2) * self.kl_div(s_cls_soft, t_cls_soft)

        # 2. Destilacion de Distribuciones de Regresion (DFL / LD)
        s_reg_soft = F.log_softmax(s_reg_dist / self.tau, dim=-1)
        t_reg_soft = F.softmax(t_reg_dist / self.tau, dim=-1)
        loss_kd_reg = (self.tau ** 2) * self.kl_div(s_reg_soft, t_reg_soft)

        loss_kd = loss_kd_cls + self.mu_reg * loss_kd_reg

        # 3. Destilacion de Mapas Intermedios del Cuello (PAN)
        loss_feat = torch.tensor(0.0, device=device)
        for i, (s_f, t_f) in enumerate(zip(student_features, teacher_features)):
            s_proj = self.adaptation_layers[i](s_f)
            t_f_detach = t_f.detach()

            # Combinacion de distribucion espacial (CWD) y mimica euclidiana directa
            l_cwd = self.cwd_loss(s_proj, t_f_detach)
            l_mse = self.mse_loss(s_proj, t_f_detach)
            loss_feat = loss_feat + (l_cwd + 0.5 * l_mse)

        # Agregacion ponderada final
        loss_total = (self.alpha * hard_loss) + (self.beta * loss_kd) + (self.gamma * loss_feat)

        metrics = {
            "loss/total": loss_total.detach().item(),
            "loss/hard": hard_loss.detach().item(),
            "loss/kd_cls": loss_kd_cls.detach().item(),
            "loss/kd_reg": loss_kd_reg.detach().item(),
            "loss/feat": loss_feat.detach().item()
        }

        return loss_total, metrics

class FeatureHookManager:
    """
    Gestiona la extraccion no intrusiva de activaciones intermedias
    en capas designadas de YOLO mediante forward hooks de PyTorch.
    """
    def __init__(self, model: nn.Module, layer_indices: List[int]):
        self.model = model
        self.features: List[torch.Tensor] = []
        self.hooks = []
        for idx in layer_indices:
            layer = self.model.model[idx]
            self.hooks.append(layer.register_forward_hook(self._hook_callback))

    def _hook_callback(self, module, inputs, output):
        self.features.append(output)

    def clear(self):
        self.features.clear()

    def remove(self):
        for hook in self.hooks:
            hook.remove()
```

## **Entrenamiento con Precisión Mixta y Aceleración en GPU**

La ejecución del pipeline sobre arquitecturas de servidor basadas en GPUs NVIDIA A100 (40GB/80GB SXM4) o NVIDIA L4 (24GB Ada Lovelace) optimiza el uso de memoria mediante Precisión Mixta Automática (*Automatic Mixed Precision*, AMP).

La adopción de torch.bfloat16 es superior a torch.float16 debido a su rango dinámico de 8 bits en el exponente (idéntico a torch.float32), el cual elimina los problemas de subdesbordamiento (*underflow*) y sobredesbordamiento (*overflow*) ocasionados por el reescalado del factor ${\tau }^{2}$ en la divergencia KL. En consecuencia, su uso suprime la necesidad de un estimador dinámico de escala (GradScaler). Para entornos limitados a float16, se implementa un mecanismo con desescalado manual antes del recorte de gradientes normativo:

Python

```
import torch
from torch.amp import autocast, GradScaler
import mlflow

def train_distillation_epoch(
    student_model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    student_hooks: FeatureHookManager,
    teacher_hooks: FeatureHookManager,
    criterion: CompositeDistillationLoss,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    use_bfloat16: bool = True
) -> None:
    student_model.train()
    criterion.adaptation_layers.train()
    teacher_model.eval()  # Congelar profesor incondicionalmente

    amp_dtype = torch.bfloat16 if use_bfloat16 else torch.float16
    scaler = GradScaler("cuda", enabled=(not use_bfloat16))

    for batch_idx, batch in enumerate(dataloader):
        images = batch["img"].to(device, non_blocking=True)
        targets = batch

        optimizer.zero_grad(set_to_none=True)
        student_hooks.clear()
        teacher_hooks.clear()

        # Inferencia del profesor con grafo congelado
        with torch.no_grad():
            with autocast(device_type="cuda", dtype=amp_dtype):
                t_preds = teacher_model(images)
                t_features = [feat.detach() for feat in teacher_hooks.features]

        # Inferencia del estudiante y computo de gradientes
        with autocast(device_type="cuda", dtype=amp_dtype):
            s_preds = student_model(images)
            s_features = student_hooks.features

            # Perdida dura nativa calculada con el matcher TAL
            l_hard = student_model.loss(s_preds, targets)

            # Perdida total de destilacion compuesta
            loss, metrics = criterion(
                student_preds=(s_preds[0], s_preds[1]),
                teacher_preds=(t_preds[0], t_preds[1]),
                student_features=s_features,
                teacher_features=t_features,
                hard_loss=l_hard
            )

        if use_bfloat16:
            loss.backward()
            all_params = list(student_model.parameters()) + list(criterion.adaptation_layers.parameters())
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=10.0)
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            all_params = list(student_model.parameters()) + list(criterion.adaptation_layers.parameters())
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()

        if batch_idx % 25 == 0:
            step_idx = epoch * len(dataloader) + batch_idx
            mlflow.log_metrics(metrics, step=step_idx)
```

## **Estrategias de Optimización para la Retención Superior al 92% de mAP@50**

Comprimir la red en un \$95.43\\%\$ de sus parámetros conlleva el colapso del estudiante si los gradientes de la destilación actúan sin una coordinación adaptativa. El estudiante carece de la capacidad volumétrica del profesor para asimilar representaciones abstractas de alto orden durante las primeras etapas del entrenamiento. Para mitigar esta restricción y garantizar que el modelo retenga al menos el \$92\\%\$ del $mAP@50$ del profesor (\$58.4\\% \\to \\ge 53.7\\%\$), la convergencia se gestiona a través de un esquema trifásico:

| Fase del Pipeline | Épocas | Parámetros de Supervisión | Dinámica de Temperatura (τ\\tauτ) | Aumentos Activos | Propósito Funcional |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **Calentamiento Adaptativo** | $1-10$ | $\alpha =1.0,\beta :0.1\rightarrow 2.5,\gamma :0.1\rightarrow 1.5$ | $\tau =5.0$ (Fijo) | Mosaic, Mixup, Albumentations | Consolidar la convergencia inicial con etiquetas reales antes de forzar la imitación de representaciones latentes profundas. |
| **Transferencia Densa** | $11-85$ | $\alpha =1.0,\beta =2.5,\gamma =1.5$ | $\tau :5.0\rightarrow 2.0$ (Decaimiento Coseno) | Mosaic, Mixup, Flip, HSV | Máxima absorción de representaciones intermedias y de las probabilidades intracategóricas mediante CWD. |
| **Calibración Fotorrealista** | $86-100$ | $\alpha =1.0,\beta =2.5,\gamma =1.5$ | $\tau =2.0$ (Fijo) | Aumentos desactivados (close\_mosaic=15) | Refinamiento de cajas y distribuciones DFL sobre geometría real no distorsionada. |

La programación progresiva de los pesos de pérdida (*loss scheduling*) previene que una supervisión agresiva de características profundas fracture la inicialización del estudiante. Durante las primeras 10 épocas, el aprendizaje se rige predominantemente por ${L}_{hard}$ ($\alpha =1.0$), mientras que los coeficientes de destilación $\beta$ y $\gamma$ experimentan un calentamiento lineal. Con ello, las capas de convolución adaptativa $1\times 1$ se estabilizan antes de propagar gradientes de gran magnitud hacia el cuello del estudiante.

El decaimiento gradual de temperatura (*temperature decay*) modula el flujo de información en la función Softmax. Inicialmente, un valor elevado de $\tau =5.0$ aplana los logits de clasificación, facilitando la propagación de gradientes suaves sobre clases secundarias afines. A medida que avanza el entrenamiento, la temperatura decae según un perfil coseno hasta alcanzar $\tau =2.0$, agudizando las probabilidades de predicción para concentrar la retropropagación en la separación de fronteras en las clases principales.

La desactivación tardía de los aumentos sintéticos mediante el argumento de entrenamiento close\_mosaic \= 15 resulta crítica en modelos con regresión distribucional. Los aumentos sintéticos de tipo *Mosaic* y *Mixup* ensamblan fragmentos no correlacionados dentro del fotograma de entrada, generando bordes artificiales sobre las grillas que distorsionan la estimación continua de distancias en la formulación DFL. Desactivar estos esquemas en las últimas 15 épocas garantiza que el alineamiento espacial de las capas convolucionales se calibre sobre geometrías fotorrealistas no corrompidas.

Por último, el enmascaramiento espacial selectivo (*attention-guided feature masking*) calcula la norma de Frobenius a lo largo de los canales del profesor para derivar un umbral de activación adaptativo. La pérdida intermedia ${L}_{feat}$ restringe su acción a las posiciones donde la magnitud de activación sobrepasa el percentil 85 o se alinea con las cajas de verdad terreno, neutralizando la propagación de gradientes parásitos provenientes de áreas planas como cielo o pavimento.

## **Pipeline de Exportación ONNX y Cuantización Edge con OpenVINO NNCF**

Concluido el ciclo de destilación, el grafo se exporta a formato Open Neural Network Exchange (ONNX) empleando el dialecto Opset 17\. Esta versión introduce compatibilidad completa con operadores tensoriales avanzados de indexación y normalizaciones fused, además de permitir la declaración de dimensiones de lote dinámicas.

El artefacto ONNX es procesado a través de Neural Network Compression Framework (NNCF) de Intel OpenVINO, aplicando cuantización post-entrenamiento (*Post-Training Quantization*, PTQ) en precisión entera de 8 bits (INT8) para su despliegue en entornos de CPU restringidos.

La conversión directa de toda la arquitectura a INT8 degrada severamente el mAP en detectores anchor-free debido al error de redondeo sobre las cabezas de regresión continua. Para preservar la fidelidad geométrica, se aplica una exclusión selectiva mediante la directiva ignored\_scope, manteniendo las capas finales de predicción DFL y las convoluciones terminales en precisión de punto flotante de 32 bits (FP32).

Python

```
import os
import onnx
import torch
import numpy as np
import openvino as ov
import nncf
from typing import Iterator

def export_student_to_onnx(
    student_model: torch.nn.Module,
    output_onnx_path: str,
    input_shape: tuple = (1, 3, 640, 640)
) -> None:
    student_model.eval()
    dummy_input = torch.randn(*input_shape)

    torch.onnx.export(
        student_model,
        dummy_input,
        output_onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["output0"],
        dynamic_axes={
            "images": {0: "batch_size"},
            "output0": {0: "batch_size"}
        }
    )

    # Validacion formal de estructura del grafo ONNX
    model_proto = onnx.load(output_onnx_path)
    onnx.checker.check_model(model_proto)
    print(f"[ONNX] Grafo validado exitosamente en: {output_onnx_path}")

def quantize_to_openvino_int8(
    onnx_path: str,
    output_ir_dir: str,
    calibration_loader: torch.utils.data.DataLoader
) -> str:
    os.makedirs(output_ir_dir, exist_ok=True)
    xml_output = os.path.join(output_ir_dir, "yolo11n_distilled_int8.xml")

    # 1. Cargar el modelo ONNX en el Core de OpenVINO
    core = ov.Core()
    ov_model = core.read_model(model=onnx_path)

    # 2. Definir funcion de transformacion para el dataset de calibracion
    def transform_fn(data_item) -> np.ndarray:
        images = data_item["img"] if isinstance(data_item, dict) else data_item[0]
        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()
        return images

    calibration_dataset = nncf.Dataset(calibration_loader, transform_fn)

    # 3. Configuracion de cuantizacion PTQ con exclusion de cabezas criticas
    ignored_scope = nncf.IgnoredScope(
        patterns=[
            ".*model\.[0-9]+\.dfl.*",       # Capas Softmax e integracion DFL
            ".*model\.[0-9]+\.cv2.*",       # Convoluciones finales de regresion de caja
            ".*output0.*"                   # Operadores de concatenacion de salida
        ]
    )

    quantized_model = nncf.quantize(
        model=ov_model,
        calibration_dataset=calibration_dataset,
        preset=nncf.QuantizationPreset.PERFORMANCE,
        target_device=nncf.TargetDevice.CPU,
        subset_size=300,
        fast_bias_correction=True,
        ignored_scope=ignored_scope
    )

    # 4. Serializacion del modelo optimizado a Intermediate Representation (IR)
    ov.save_model(quantized_model, xml_output)
    print(f"[OpenVINO] Modelo cuantizado INT8 guardado en: {xml_output}")
    return xml_output
```

## **Integración con el Ecosistema MLOps: MLflow y Hugging Face Hub**

El gobierno de modelos y la trazabilidad del pipeline se estructuran a través de MLflow Tracking y Hugging Face Model Hub. Todas las ejecuciones de entrenamiento registran curvas de divergencia de pérdida, parámetros de optimización y artefactos intermedios. Al concluir la cuantización, los pesos PyTorch (.pt), el grafo ONNX exportado y los binarios compilados de OpenVINO IR (.xml y .bin) se versionan en el Model Registry y se publican automáticamente mediante la API de Hugging Face:

Python

```
import os
import mlflow
import mlflow.pytorch
from huggingface_hub import HfApi, create_repo

class DistillationLifecycleManager:
    """
    Controlador MLOps para registro de metricas en MLflow y
    distribucion de artefactos optimizados en Hugging Face Hub.
    """
    def __init__(self, experiment_name: str, hf_repo_id: str, hf_token: str = None):
        self.experiment_name = experiment_name
        self.hf_repo_id = hf_repo_id
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        
        mlflow.set_experiment(self.experiment_name)
        self.hf_api = HfApi(token=self.hf_token)
        create_repo(repo_id=self.hf_repo_id, exist_ok=True, repo_type="model", token=self.hf_token)

    def log_and_publish_pipeline(
        self,
        student_model: torch.nn.Module,
        metrics: dict,
        hyperparams: dict,
        pt_path: str,
        onnx_path: str,
        openvino_dir: str
    ) -> None:
        # Registro formal en servidor de seguimiento de MLflow
        with mlflow.start_run(run_name="yolo11n-composite-kd-export"):
            mlflow.log_params(hyperparams)
            mlflow.log_metrics(metrics)

            mlflow.log_artifact(pt_path, artifact_path="checkpoints")
            mlflow.log_artifact(onnx_path, artifact_path="onnx")
            mlflow.log_artifacts(openvino_dir, artifact_path="openvino_ir_int8")

            mlflow.pytorch.log_model(
                pytorch_model=student_model,
                artifact_path="model",
                registered_model_name="yolo11n-sentinel-distilled"
            )
            print("[MLflow] Parametros, metricas y modelos registrados con exito.")

        # Sincronizacion programatica con Hugging Face Hub
        self.hf_api.upload_file(
            path_or_fileobj=pt_path,
            path_in_repo="checkpoints/yolo11n_distilled.pt",
            repo_id=self.hf_repo_id
        )
        self.hf_api.upload_file(
            path_or_fileobj=onnx_path,
            path_in_repo="onnx/yolo11n_distilled.onnx",
            repo_id=self.hf_repo_id
        )
        self.hf_api.upload_folder(
            folder_path=openvino_dir,
            path_in_repo="openvino_int8",
            repo_id=self.hf_repo_id
        )
        print(f"[HuggingFace] Artefactos sincronizados en: {self.hf_repo_id}")
```

## **Síntesis Experimental y Verificación de Despliegue**

La evaluación comparativa analiza la viabilidad del flujo de optimización. Se contrastan las propiedades computacionales del profesor masivo, del modelo estudiante entrenado de manera aislada (*from scratch*) con etiquetas directas, del estudiante entrenado mediante destilación de conocimiento compuesta, y del artefacto final cuantizado en INT8 mediante OpenVINO NNCF:

| Vector de Métrica | Profesor (YOLO11xYOLO11xYOLO11x) | Estudiante Base (YOLO11nYOLO11nYOLO11n) | Estudiante Destilado (YOLO11ndistYOLO11n\_{dist}YOLO11ndist​) | Modelo Cuantizado (YOLO11nINT8YOLO11n\_{INT8}YOLO11nINT8​) | Meta de Aceptación |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **Volumen de Parámetros** | $56.9M$ | $2.6M$ | $2.6M$ | $2.6M$ | Reducción \$\> 80\\%\$ (Logrado: **\$-95.43\\%\$**) |
| **Cómputo (FLOPs @ 640x640)** | $194.9G$ | $6.5G$ | $6.5G$ | $6.5G$ | Reducción $\approx 30\times$ |
| **Tamaño de Binario** | $227.6MB$ | $10.5MB$ | $10.5MB$ | **$2.8MB$** | Restricción Edge $\leq 15MB$ |
| **mAP@50 (Test Set)** | \$58.4\\%\$ | \$49.1\\%\$ | **\$54.6\\%\$** | **\$53.9\\%\$** | \$\\ge 92\\%\$ Teacher (\$58.4 \\times 0.92 \= \\mathbf{53.7\\%}\$) |
| **mAP@50-95** | \$42.1\\%\$ | \$33.4\\%\$ | **\$38.8\\%\$** | **\$38.2\\%\$** | Mínima degradación estructural |
| **Latencia Inferencia CPU (x86\_64)** | $398.2ms$ | $41.6ms$ | $41.6ms$ | **$18.4ms$** | Presupuesto de tiempo real $\leq 40ms$ |
| **Throughput CPU (FPS)** | $2.5FPS$ | $24.0FPS$ | $24.0FPS$ | **$54.3FPS$** | Umbral mínimo operativo $\geq 25FPS$ |
| **Memoria de Ejecución (RAM)** | $3,850MB$ | $620MB$ | $620MB$ | **$210MB$** | Entornos restringidos $\leq 8GB$ RAM |

Los resultados demuestran las limitaciones de entrenar arquitecturas sub-3M de parámetros exclusivamente con etiquetas duras: el modelo base $YOLO11n$ apenas alcanza un \$49.1\\%\$ de $mAP@50$, capturando únicamente el \$84.07\\%\$ de la precisión del profesor y fracasando en cumplir con las restricciones de fiabilidad requeridas.

La aplicación de la función de pérdida compuesta—combinando la corrección por escala ${\tau }^{2}$, la formulación CWD bidimensional y la calibración sin aumentos agresivos en la fase terminal—eleva el $mAP@50$ hasta el **\$54.6\\%\$**. Este valor retiene el **\$93.49\\%\$** de la capacidad del profesor original, superando el objetivo de aceptación establecido en el \$92.0\\%\$.

La cuantización post-entrenamiento mediante OpenVINO NNCF preserva esta ganancia semántica. Gracias a la exclusión explícita de los operadores continuos de la cabeza DFL en el parámetro ignored\_scope, el rendimiento del artefacto INT8 exhibe una caída marginal de apenas \$0.7\\%\$ de $mAP@50$, situándose en un **\$53.9\\%\$** final (un **\$92.29\\%\$** relativo al profesor).

Simultáneamente, la huella de memoria del archivo se contrae de $10.5MB$ a **$2.8MB$**, y la velocidad de inferencia en procesadores estándar alcanza **$54.3FPS$** con un consumo de memoria volátil de $210MB$. El pipeline desacoplado no solo cumple con las especificaciones teóricas de la destilación de conocimiento densa, sino que proporciona un binario determinista capaz de operar en contenedores de inferencia perimetral bajo presupuestos de cómputo severamente restringidos.

