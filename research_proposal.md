# TwinMask: Joint Masked Diffusion over Text and Topology for Text-Attributed Graphs

> پروپوزال پژوهشی — نسخه ۱ (نوامبر ۲۰۲۶)
> از «ایده خام Joint Text-Graph Diffusion» به یک مسئله پژوهشی با ادعای نوآوری مشخص

---

## ۰. خلاصه در یک پاراگراف

در Text-Attributed Graph (TAG) هر Node یک متن خام دارد و Edgeها ساختار را می‌سازند. **هم توکن‌های متن و هم Edgeها گسسته‌اند**؛ پس می‌توان هر دو را با *یک* فرآیند Diffusion گسسته از نوع Masking (Absorbing-state، مثل MDLM/LLaDA) تخریب کرد و با *یک* Denoiser بازسازی کرد. نوآوری ما سه بخش دارد:
1. **نرخ نویز جدا برای هر Modality** ($t_x$ برای متن، $t_a$ برای Edgeها) تا یک مدل واحد همه توزیع‌ها را یاد بگیرد: $p(A\mid X)$ (Link Prediction / Cold-start)، $p(X\mid A)$ (تکمیل متن Nodeها)، $p(X,A)$ (تولید TAG) و نمایش Nodeها.
2. **Topology Mask پویا**: ساختار گراف هم‌زمان *ورودی* Attention و *خروجی* Denoising است؛ با Unmask شدن Edgeها، مسیر Message Passing بین متن‌ها تغییر می‌کند (متن و ساختار هم‌زمان تکامل می‌یابند).
3. **یک ابزار تحلیلی**: از همان مدل، تخمینی از اطلاعات متقابل متن و ساختار (Text–Structure MI) به‌دست می‌آید که پیش‌بینی می‌کند *چه زمانی* Cross-modal Denoising واقعاً کمک می‌کند (مثلاً گراف‌های Homophilic در برابر Heterophilic).

---

## ۱. نقد صادقانه ایده خام

ایده اولیه (Gaussian Diffusion روی Embedding متن + Edge Corruption + GNN مشترک + Consistency Loss) جهت درستی دارد، اما در شکل فعلی **ادعای نوآوری ضعیفی** دارد:

| مؤلفه ایده خام | کار قبلی که عملاً همین را انجام داده |
|---|---|
| Diffusion مشترک ویژگی Node و ماتریس مجاورت | **GDSS** (Jo et al., ICML 2022)، **DiGress** (Vignac et al., ICLR 2023)، **GraphMaker** (Li et al.) که دقیقاً گراف‌های Attributed بزرگ مثل Cora را تولید می‌کند |
| بازسازی ویژگی نویزی با کمک همسایه‌ها | **GraphMAE** (KDD 2022)؛ و برای Diffusion: **DDM** (NeurIPS 2023)، **Graffe** (2025) |
| حذف Edge و بازسازی آن با GNN | **MaskGAE** (Li et al., KDD 2023) |
| Diffusion روی Embedding متن | **Diffusion-LM** (Li et al., 2022) — و خود ایده خام اذعان می‌کند که این «DLM واقعی» نیست |
| $\mathcal{L}_{cross}$ (سازگاری بازسازی با ساختار) | یک Regularizer سازگاری؛ به‌تنهایی مشارکت علمی حساب نمی‌شود |

یک یافته مهم از ادبیات: **GraphMaker گزارش می‌کند که Denoising هم‌زمان ساختار و ویژگی روی گراف‌های Attributed بزرگ بد عمل می‌کند** و به‌همین دلیل Denoising را *ناهمزمان* (Asynchronous) انجام می‌دهد. یعنی فرض «Joint همیشه بهتر است» خودش محل سؤال است — و این دقیقاً فرصتی برای ماست (بخش ۳.۲).

**نتیجه:** نسخه Embedding-level را فقط به‌عنوان Baseline/Sanity-check نگه می‌داریم، نه به‌عنوان مشارکت اصلی.

---

## ۲. مرور ادبیات (نقشه حوزه)

### ۲.۱ Diffusion Language Models (DLM)
- **D3PM** (Austin et al., 2021): چارچوب Diffusion گسسته با ماتریس‌های انتقال، از جمله Absorbing-state ([MASK]).
- **Diffusion-LM** (Li et al., 2022): Diffusion پیوسته در فضای Embedding توکن‌ها.
- **SEDD** (Lou et al., ICML 2024) و **MDLM** (Sahoo et al., NeurIPS 2024): نشان دادند Masked Diffusion با یک ELBO ساده (Cross-entropy وزن‌دار روی توکن‌های Mask‌شده) رقیب AR است.
- **LLaDA** (Nie et al., 2025): Masked DLM در مقیاس ۸B.
- **UniDisc** (Swerdlow et al., 2025): Masked Diffusion مشترک متن+تصویر در یک Transformer دوطرفه؛ برای CFG، با احتمال ۰.۱ کل یک Modality را Mask می‌کند.
- **UniDiffuser** (Bao et al., ICML 2023): ایده کلیدی برای ما — **Timestep جدا برای هر Modality**؛ $t=0$ یعنی شرط‌گذاری، $t=T$ یعنی Marginal. یک شبکه همه توزیع‌های Joint/Conditional/Marginal را یاد می‌گیرد.
- **Diffusion Forcing** (Chen et al., NeurIPS 2024): سطح نویز مستقل برای هر توکن.

### ۲.۲ Diffusion برای گراف
- تولید گراف: **EDP-GNN**، **GDSS** (پیوسته، X و A مشترک)، **DiGress** (گسسته، Node/Edge)، **EDGE** (Edge-centric، مقیاس‌پذیر)، **GraphMaker** (گراف Attributed بزرگ، Edge Mini-batching، Denoising ناهمزمان)، **LGD** (Latent Graph Diffusion, NeurIPS 2024؛ یکپارچه‌سازی تولید و پیش‌بینی).
- یادگیری نمایش با Diffusion: **DDM** (نویز جهت‌دار و Anisotropic)، **Graffe** (2025؛ Encoder گراف به‌عنوان شرط Decoder Diffusion، با کران پایین روی Conditional MI)، **DiffGraph** (Heterogeneous, Cross-view Denoising).
- Link Prediction با Diffusion: **Sub-graph Based Diffusion for Link Prediction** (arXiv 2409.08487).

### ۲.۳ مدل‌های زبانی روی TAG
- LM به‌عنوان Encoder/Enhancer: **GIANT**، **GLEM** (ICLR 2023)، **TAPE** (He et al., ICLR 2024؛ متن خام Cora/PubMed را منتشر کرد)، **SimTeG**.
- LLM به‌عنوان Predictor: **GraphGPT**، **LLaGA**، **InstructGLM**، **LinkGPT**.
- **TAG-DLM** (Chen, Bei, …, Tong — arXiv 2606.31166، ژوئن ۲۰۲۶) — **نزدیک‌ترین کار به ما**:
  - LLaDA-8B-Instruct فریز + LoRA؛ همسایگی Node خطی‌سازی شده و ساختار از طریق **Topology Attention Mask ثابت** تزریق می‌شود.
  - **فقط توکن پاسخ (Label یا Yes/No) Mask می‌شود.** متن Nodeها Diffuse نمی‌شود و **Edgeها نه Diffuse و نه تولید می‌شوند.**
  - یعنی از DLM فقط به‌عنوان Classifier دوطرفه استفاده می‌کند؛ ظرفیت *مولد* و *Joint* آن بلااستفاده می‌ماند.
  - دیتاست‌ها: Cora، PubMed، ogbn-arxiv.

### ۲.۴ تولید TAG (متن + ساختار)
- **UTGDiff** (arXiv 2408.09896): Masked Diffusion متن+گراف در یک Transformer، اما برای **مولکول**؛ متن فقط *شرط* است و Nodeها متن ندارند.
- **Controllable TAG Generation via LLM-Based Substructure Modeling** (Mao et al., EMNLP 2026): اولین مدل آموزش‌دیده برای توزیع Joint متن خام + Edge + Label — اما **Autoregressive**؛ زیرگراف با پیمایش اویلری سریال می‌شود و ترتیب پیمایش به‌عنوان متغیر نهان در یک کران پایین Variational وارد می‌شود. خود این مقاله می‌گوید روش‌های قبلی یا متن را با Embedding جایگزین می‌کنند یا فقط Prompting/Agent هستند.
- **GraphMaster**، **GAG**، **TANS**، **LLM4NG**: مبتنی بر Prompting/Agent؛ توزیع Joint را یاد نمی‌گیرند.

### ۲.۵ شکاف پژوهشی (Gap)

| روش | متن خام | Diffusion روی متن | Diffusion/تولید Edge | Conditional در هر دو جهت | مستقل از ترتیب |
|---|:-:|:-:|:-:|:-:|:-:|
| GDSS / DiGress / GraphMaker | ✗ (بردار) | — | ✓ | ✗ | ✓ |
| GraphMAE / MaskGAE / DDM | ✗ | ✗ | جزئی | ✗ | ✓ |
| TAG-DLM | ✓ | ✗ (فقط پاسخ) | ✗ | ✗ | ✓ |
| UTGDiff | متن فقط شرط | ✗ | ✓ (مولکول) | ✗ | ✓ |
| Mao et al. (EMNLP 2026) | ✓ | ✗ (AR) | ✓ | از طریق Prefix | ✗ (پیمایش اویلری) |
| **TwinMask (ما)** | ✓ | ✓ | ✓ | ✓ | ✓ |

> **هیچ کار شناخته‌شده‌ای یک Masked Diffusion واحد روی توکن‌های متن خام Nodeها و متغیرهای Edge با سطح نویز جدا برای هر Modality ارائه نکرده است.**
> (این ادعا باید پیش از نگارش مقاله با جست‌وجوی دوباره arXiv به‌روز شود؛ حوزه بسیار سریع حرکت می‌کند.)

---

## ۳. روش پیشنهادی: TwinMask

### ۳.۱ صورت‌بندی

برای یک زیرگراف نمونه‌برداری‌شده $S$ با $k$ Node (مثلاً Ego-graph دوگامی، $k\le 32$):
- متن Node $i$: دنباله توکن $s_i=(s_i^1,\dots,s_i^L)$ (عنوان + ابتدای چکیده، $L\approx 64$).
- متغیر Edge برای هر جفت در مجموعه کاندید $C\subseteq S\times S$: $e_{ij}\in\{0,1\}$. ($C$ = همه جفت‌های داخل $S$، یا مثبت‌ها + نمونه منفی.)

**فرآیند رو به جلو (دو ساعت مستقل):** با Schedule خطی $\alpha(t)=1-t$:
$$
q(z_{s_i^l}\mid s_i^l)=\alpha(t_x)\,\delta_{s_i^l}+(1-\alpha(t_x))\,\delta_{[M]},\qquad
q(z_{e_{ij}}\mid e_{ij})=\alpha(t_a)\,\delta_{e_{ij}}+(1-\alpha(t_a))\,\delta_{[M]}
$$
یک Edge Mask‌شده یعنی «نامعلوم» — **نه** «وجود ندارد». این تفاوت مهمی با Edge Dropout در ایده خام است (آنجا حذف Edge با «عدم وجود» اشتباه گرفته می‌شود).

**نمونه‌برداری زمان‌ها در آموزش:** $(t_x,t_a)\sim U[0,1]^2$ مستقل، به‌اضافه با احتمال کوچک $p$ حالت‌های مرزی: $t_x=0$ (متن کامل → $p(A\mid X)$)، $t_a=0$ (گراف کامل → $p(X\mid A)$)، $t_x=1$ یا $t_a=1$ (Marginal → CFG رایگان).

**تابع هزینه (MDLM-style، دو Modality):**
$$
\mathcal{L}(\theta)=\mathbb{E}_{t_x,t_a}\Big[\tfrac{1}{t_x}\!\!\sum_{(i,l):\,z_{s_i^l}=[M]}\!\!-\log p_\theta(s_i^l\mid z)\;+\;\lambda\,\tfrac{1}{t_a}\!\!\sum_{(i,j):\,z_{e_{ij}}=[M]}\!\!-\log p_\theta(e_{ij}\mid z)\Big]
$$
که $z$ کل حالت نویزی (متن+Edge) است.

**نکته نظری (قابل اثبات در مقاله):** برای هر مسیر یکنوا $\tau\mapsto(t_x(\tau),t_a(\tau))$ از $(0,0)$ تا $(1,1)$، NELBO پیوسته برابر است با
$$
\int_0^1\Big[\tfrac{t_x'(\tau)}{t_x(\tau)}\,\mathbb{E}\!\sum_{\text{masked text}}\!-\log p_\theta+\tfrac{t_a'(\tau)}{t_a(\tau)}\,\mathbb{E}\!\sum_{\text{masked edges}}\!-\log p_\theta\Big]d\tau .
$$
چون Denoiser روی کل مربع $[0,1]^2$ آموزش دیده، **ترتیب تولید (اول متن، اول گراف، یا هم‌زمان) یک انتخاب در زمان Inference است**، نه یک Inductive bias ثابت. این پاسخ اصولی به یافته GraphMaker است: به‌جای اینکه ناهمزمانی را دستی انتخاب کنیم، مسیر بهینه را یاد می‌گیریم/جست‌وجو می‌کنیم.

### ۳.۲ Denoiser: Transformer با Topology Mask پویا

- ورودی: توکن‌های متن همه Nodeهای $S$ + یک توکن خلاصه $[\mathrm{NODE}_i]$ برای هر Node + Embedding زمان‌های $t_x,t_a$ (به‌صورت توکن، مثل UniDiffuser).
- Attention درون هر Node: کامل.
- Attention بین Nodeها (فقط بین توکن‌های $[\mathrm{NODE}]$ یا همه توکن‌ها — قابل Ablation): یک **Bias** وابسته به حالت فعلی Edge:
$$
B_{ij}=\begin{cases}0 & z_{e_{ij}}=1\\ -\infty & z_{e_{ij}}=0\\ b_{[M]}(t_a) & z_{e_{ij}}=[M]\ \ (\text{قابل یادگیری})\end{cases}
$$
  در TAG-DLM این ماسک **ثابت** است؛ اینجا **خودش در حال Denoise شدن است** — هر Edge که در فرآیند Sampling آشکار شود، مسیر Message Passing را باز یا بسته می‌کند.
- Head متن: Head زبانی استاندارد روی توکن‌های Mask‌شده.
- Head Edge: $p_\theta(e_{ij}=1\mid z)=\sigma\big(\mathrm{MLP}[h_i,h_j,h_i\odot h_j,\phi(t_a)]\big)$ روی $h_i=h_{[\mathrm{NODE}_i]}$.
- Backbone: (الف) Transformer کوچک از صفر، (ب) راه‌اندازی از یک MDLM کوچک از پیش آموزش‌دیده (مثلاً چک‌پوینت ~۱۱۰M منتشرشده MDLM روی OpenWebText — در دسترس بودنش را بررسی کن)، (ج) در فاز آخر LoRA روی LLaDA-8B برای مقایسه مستقیم با TAG-DLM.

### ۳.۳ Inference: یک مدل، چند وظیفه

| وظیفه | تنظیم |
|---|---|
| Link Prediction | $t_x=0$، Edge هدف Mask، بقیه Edgeهای آموزشی مشهود؛ امتیاز = $p_\theta(e_{ij}=1)$ |
| Cold-start Node (فقط متن، بدون Edge) | همه Edgeهای Node جدید Mask؛ Unmask تکراری بر اساس اطمینان |
| تکمیل متن ناقص/مفقود | $t_a=0$، متن Node هدف (کامل یا جزئی) Mask |
| تولید TAG کامل | از $(1,1)$ به $(0,0)$ در یک مسیر انتخابی؛ Unmask بر اساس Confidence در **هر دو** Modality |
| نمایش Node | $h_i$ در $(t_x,t_a)$ کوچک → Linear Probe |
| CFG | $\tilde p\propto p(e\mid X)^{1+w}\,p(e\mid X{=}[M])^{-w}$ — رایگان چون Marginal یاد گرفته شده |

### ۳.۴ ابزار تحلیلی: Text–Structure Mutual Information (TSMI)

چون مدل هم $p(A\mid X)$ و هم $p(A)$ (با $t_x=1$) را می‌دهد:
$$
\widehat{I}(A;X)\;\approx\;\mathbb{E}\big[\mathrm{NELBO}(A\mid X{=}[M])-\mathrm{NELBO}(A\mid X)\big]
$$
(تخمین مبتنی بر تفاضل دو کران — نه مقدار دقیق؛ باید در مقاله صریحاً گفته شود.) قابل محاسبه **در سطح Node** و **در سطح دیتاست**.

**فرضیه قابل آزمون:** سود Cross-modal Denoising (نسبت به Diffusion مستقل) با $\widehat{I}$ رابطه یکنوا دارد؛ و $\widehat{I}$ روی گراف‌های Heterophilic کمتر است، بنابراین روش‌هایی که همسایگی را کورکورانه برای بازسازی متن به کار می‌برند (مثل ایده خام) در آنجا *آسیب* می‌زنند. این یک یافته علمی است، فراتر از «مدل ما بهتر است».

---

## ۴. سؤال‌ها و فرضیه‌های پژوهشی

- **RQ1:** آیا یک Masked Diffusion مشترک روی متن خام و Edgeها در Link Prediction و Node Classification با TAG-DLM و GNNهای قوی رقابت می‌کند — با اینکه یک مدل *مولد* است؟
- **RQ2:** آیا جدا کردن $t_x,t_a$ نسبت به Timestep مشترک (و نسبت به دو مدل مستقل) بهتر است؟ کدام مسیر تولید (متن‌اول/گراف‌اول/هم‌زمان) بهترین TAG را می‌سازد؟
- **RQ3:** آیا Topology Mask پویا (در برابر ثابت یا بدون Mask) لازم است؟
- **RQ4:** آیا TSMI سود Cross-modal را در دیتاست‌ها و Nodeهای مختلف پیش‌بینی می‌کند (Homophilic در برابر Heterophilic)؟
- **RQ5 (تولید):** آیا TAGهای تولیدی هم از نظر ساختاری (MMD درجه/خوشه‌بندی/طیف) و هم از نظر متن و *سازگاری متن–ساختار* با Mao et al. (AR) و GraphMaker رقابت می‌کنند — با Sampling موازی و بدون وابستگی به ترتیب پیمایش؟

---

## ۵. طرح آزمایش‌ها

### ۵.۱ دیتاست‌ها
- **Cora، PubMed** با متن خام (نسخه منتشرشده توسط TAPE).
- **ogbn-arxiv** (عنوان + چکیده) — برای مقیاس.
- **CS-TAG / TEG-DB** (مثلاً Books-History، Photo) — دامنه‌های غیر استنادی.
- **TAGهای Heterophilic** با متن خام (نسخه‌های Cornell/Texas/Wisconsin/Washington که در کارهای اخیر LLM روی Heterophily منتشر شده‌اند) — برای RQ4.

### ۵.۲ Baselineها
- GCN / GAT / SAGE روی Embedding MiniLM؛ GraphMAE؛ MaskGAE؛ TAPE یا GLEM (در صورت امکان).
- TAG-DLM (Link Prediction و Node Classification).
- GraphMaker، Mao et al. (تولید).
- **Ablationهای خودمان:** (۱) Text-only diffusion، (۲) Edge-only diffusion، (۳) Joint با $t_x=t_a$، (۴) بدون Topology Mask، (۵) Mask ثابت، (۶) نسخه Embedding-level (ایده خام).

### ۵.۳ معیارها و پروتکل
- **Link Prediction:** AUC، AP، و **Hits@K / MRR با پروتکل HeaRT** (Li et al., NeurIPS 2023 D&B) — چون Negative Sampling تصادفی نتایج را بیش‌برآورد می‌کند. Edgeهای Validation/Test از ورودی حذف شوند (Leakage).
- **Node Classification:** Accuracy، Macro-F1 با Linear Probe.
- **Cold-start:** Nodeهای Test همه Edgeهایشان حذف؛ بازیابی Edge فقط از متن.
- **تکمیل متن:** Perplexity/NELBO، BERTScore، و دقت Classifier روی متن بازسازی‌شده.
- **تولید TAG:** MMD درجه/Clustering/Spectrum؛ کیفیت متن (MAUVE یا Perplexity با یک LM مرجع)؛ Homophily معنایی گراف تولیدی؛ Train-on-Synthetic-Test-on-Real (مثل GraphMaker).
- **شکل امضایی مقاله:** Heatmap دوبعدی عملکرد روی شبکه $(t_x,t_a)$ برای هر دیتاست، کنار مقدار TSMI.

---

## ۶. نقشه اجرا (مرحله‌ای، کم‌هزینه → پرهزینه)

| فاز | محتوا | هدف «Go/No-go» | زمان تقریبی |
|---|---|---|---|
| **۰. Sanity** | نسخه Embedding-level روی Cora با $t_x,t_a$ جدا (MiniLM + Gaussian، Edge Masking با حالت [M]، GCN/Transformer کوچک) | آیا Conditioning متقابل در LP و بازسازی Embedding کمک می‌کند؟ مقایسه با GraphMAE/MaskGAE | ۲–۳ هفته |
| **۱. TwinMask-S** | Masked Diffusion روی توکن‌ها، Transformer ~۵۰–۱۱۰M، Cora/PubMed (متن TAPE)، زیرگراف $k\le 32$ | رقابت با GNNها در LP؛ تکمیل متن معنادار | ۴–۶ هفته |
| **۲. تحلیل** | Heatmap $(t_x,t_a)$، Ablationها، TSMI، دیتاست‌های Heterophilic | تأیید/رد RQ2–RQ4 | ۳–۴ هفته |
| **۳. مقیاس** | ogbn-arxiv؛ اختیاری: LoRA روی LLaDA-8B برای مقایسه مستقیم با TAG-DLM | RQ1 در مقیاس | ۴ هفته |
| **۴. تولید** | تولید TAG و مقایسه با Mao et al. و GraphMaker | RQ5 | ۳ هفته |

**نکته منابع محاسباتی:** فازهای ۰–۲ با یک GPU مصرفی (۱۲–۲۴GB) قابل انجام‌اند. توالی هر زیرگراف ≈ $32\times 64=2048$ توکن است.

### ساختار پیشنهادی کد
```
twinmask/
├── data/            # بارگذاری TAG با متن خام، Split لینک بدون Leakage، نمونه‌برداری زیرگراف
├── diffusion/       # Masking دو-ساعته، Schedule، ELBO، Samplerهای مسیرمحور
├── models/
│   ├── embedding_baseline.py   # فاز ۰ (ایده خام با t جدا)
│   ├── denoiser.py             # Transformer + Topology Bias پویا
│   └── heads.py                # Head متن و Head Edge
├── analysis/        # TSMI، Heatmap (t_x, t_a)
├── eval/            # LP (HeaRT)، NC، Cold-start، تولید
└── configs/
```

---

## ۷. ریسک‌ها و راه‌حل‌ها

| ریسک | راه‌حل |
|---|---|
| مدل کوچک از صفر متن خوبی تولید نکند | راه‌اندازی از MDLM ازپیش‌آموزش‌دیده؛ یا ادعای اصلی را روی LP/Cold-start/TSMI متمرکز کن و تولید متن را ثانویه بگیر |
| $O(k^2)$ متغیر Edge | زیرگراف کوچک + Negative Sampling داخل زیرگراف؛ برای تولید کل گراف: Edge Mini-batching مثل GraphMaker |
| سود Joint ناچیز باشد | خود این، با تحلیل TSMI، یک یافته منفی قابل انتشار است («چه زمانی Joint کمک نمی‌کند») |
| کار مشابه در ماه‌های آینده منتشر شود | فاز ۰–۱ را سریع انجام بده؛ پیش از نگارش دوباره arXiv را بررسی کن؛ سهم تحلیلی (TSMI، مسیرهای $(t_x,t_a)$) کمتر در معرض تکرار است |
| ارزیابی LP خوش‌بینانه | پروتکل HeaRT و حذف دقیق Edgeهای Test از ورودی |

---

## ۸. مشارکت‌های مورد انتظار (برای Abstract مقاله)

1. اولین **Masked Diffusion مشترک روی متن خام Nodeها و Edgeها** در TAGها — یک DLM واقعی، نه Diffusion روی Embedding.
2. **Diffusion با نرخ نویز جدا برای هر Modality** که در یک مدل، Link Prediction، Cold-start، تکمیل متن، تولید TAG و یادگیری نمایش را یکپارچه می‌کند؛ با یک خانواده ELBO مسیرمحور که ترتیب تولید را به انتخاب زمان Inference تبدیل می‌کند.
3. **Topology Attention پویا** که ساختار را هم‌زمان شرط و هدف Denoising قرار می‌دهد.
4. **TSMI**: تخمین‌گری از وابستگی متن–ساختار که توضیح می‌دهد چه وقت مدل‌سازی مشترک کمک می‌کند، با شواهد روی گراف‌های Homophilic و Heterophilic.

**مقصدهای احتمالی:** ابتدا یک Workshop (مثلاً LoG یا Workshopهای NeurIPS/ICLR)، سپس KDD / ICLR / LoG main.

---

## ۹. منابع کلیدی برای مطالعه (به ترتیب اولویت)

1. TAG-DLM — https://arxiv.org/abs/2606.31166
2. Mao et al., Controllable TAG Generation via LLM-Based Substructure Modeling (EMNLP 2026) — https://aclanthology.org/2026.emnlp-main.996.pdf
3. MDLM (Sahoo et al., NeurIPS 2024) و LLaDA (Nie et al., 2025)
4. UniDiffuser — https://arxiv.org/abs/2303.06555 ؛ UniDisc — https://arxiv.org/abs/2503.20853
5. GraphMaker — https://arxiv.org/abs/2310.13833
6. UTGDiff — https://arxiv.org/abs/2408.09896
7. DiGress (ICLR 2023)، GDSS (ICML 2022)، EDGE (ICML 2023)
8. GraphMAE (KDD 2022)، MaskGAE — https://arxiv.org/abs/2205.10053
9. DDM — https://arxiv.org/abs/2306.13210 ؛ Graffe — https://arxiv.org/abs/2505.04956
10. TAPE (ICLR 2024)، GLEM (ICLR 2023)، LLaGA (ICML 2024)
11. What Gets Unmasked First? Graph-to-Text با DLM — https://arxiv.org/abs/2605.31564
12. HeaRT: Evaluating GNNs for Link Prediction (NeurIPS 2023 D&B)
