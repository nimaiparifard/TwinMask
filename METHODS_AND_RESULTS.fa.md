# گزارش روش‌های پیاده‌سازی‌شده و نتایج پروژه

این گزارش بر اساس کدها و خروجی‌های ذخیره‌شده در مخزن تهیه شده است. آزمایش جدیدی اجرا نشده است. جدول‌های بخش بعد مستقیماً از JSONها استخراج شده‌اند؛ گزارش‌های تفصیلی موجود نیز در ادامه ضمیمه شده‌اند.

## توضیح روش‌ها

### ۱. TwinMask در سطح بردار — فاز صفر

در `twinmask/phase0.py` متن نودها با MiniLM به بردار تبدیل می‌شود. روی بردار متن نویز Gaussian و روی یال‌ها ماسک اعمال می‌شود و یک denoiser مبتنی بر GCN بازسازی را یاد می‌گیرد. نمایش‌ها بدون برچسب کلاس آموزش می‌بینند.

در نسخهٔ joint شدت نویز متن و یال مستقل است؛ در tied یک شدت مشترک دارند. text_only فقط متن را خراب می‌کند و edge_only فقط یال را. این نسخه‌ها برای تفکیک اثر دو مدالیته مقایسه می‌شوند. در نویز شدید، کمک ساختار به بازسازی متن در joint بیشتر از tied است؛ مثلاً افزایش cosine در Cora برابر ۰٫۰۸۹ در مقابل ۰٫۰۳۹ است. این مزیت به برتری عمومی در تمام وظایف تبدیل نمی‌شود: MRR لینک‌های سخت Cora برای joint برابر ۱۴٫۵ و برای شباهت متن ۲۰٫۶ درصد است.

### ۲. TwinMask در سطح توکن — فازهای یک تا سه

در `twinmask/model.py` و `twinmask/tm.py`، DistilRoBERTa توکن‌های متن را پردازش می‌کند و GraphMixer بین نودها اطلاعات منتقل می‌کند. توکن‌ها و یال‌ها با شدت مستقل ماسک می‌شوند. حالت یال نامعلوم با یال غایب متفاوت است و وضعیت جاری یال بر attention اثر می‌گذارد. تابع آموزش از cross-entropy روی اقلام ماسک‌شده استفاده می‌کند.

نسخه‌های joint، tied، text_only و edge_only با دو کنترل دیگر مقایسه می‌شوند: static ماسک یال را مانند نبود یال می‌گیرد و notopo اطلاعات توپولوژی را از mixer حذف می‌کند. فاز سه همین معماری را روی ogbn-arxiv اجرا می‌کند.

نتیجهٔ مهم، مقاومت بهتر در نبود متن است. در Cora، AUC ساختارمحور joint برابر ۶۴٫۵ و edge_only برابر ۴۲٫۵ درصد است. در PubMed این اعداد ۸۶٫۹ و ۷۸٫۴ هستند. با این حال، در PubMed، MRR نسخهٔ edge_only برابر ۲۴٫۶ و joint برابر ۱۸٫۴ است؛ بنابراین معیار و وضعیت ورودی در انتخاب روش اهمیت دارند. در سطح توکن، برتری همیشگی joint بر tied تأیید نشده است.

در ablation توپولوژی PubMed، AUC ساختارمحور joint برابر ۸۶٫۹، static برابر ۸۱٫۹ و notopo برابر ۷۹٫۷ است. در مقابل، static در رتبه‌بندی منفی‌های سخت MRR بهتری دارد. در arxiv، joint به AUC لینک ۹۸٫۱ و AUC بدون متن ۸۸٫۴ درصد می‌رسد. بیشتر آزمایش‌های این فازها تک-seed هستند.

### ۳. تحلیل TSMI و بازسازی متن

`twinmask/phase2.py` با اختلاف تخمین NELBO اثر متن بر یال، یال بر متن و همسایگی بر متن را اندازه می‌گیرد و تغییر رفتار در شدت‌های مختلف نویز را بررسی می‌کند. این بخش ابزار تحلیل است، نه مدل طبقه‌بندی جداگانه. اختلاف دو کران NELBO را نباید mutual information دقیق تلقی کرد.

کمک همسایگی به بازسازی متن در Cora، CiteSeer و PubMed مثبت و در WikiCS، Instagram و Reddit منفی گزارش شده است؛ بنابراین استفاده از همسایه‌ها همیشه مفید نیست.

### ۴. پل DIVERGE و ensemble نماهای diffusion

در `scripts/diverge_bridge.py` و `scripts/tune_bridge.py` برای embedding خام و نمایش‌های diffusion، طبقه‌بند SAGE جداگانه آموزش داده می‌شود. خروجی‌ها با میانگین احتمال یا WeightLearner ترکیب می‌شوند. نمایش diffusion بدون برچسب کلاس است، اما طبقه‌بند و وزن ensemble از برچسب استفاده می‌کنند. نسخهٔ تیون‌شده باید از آزمایش اولیهٔ بدون تیونینگ جدا خوانده شود.

در گزارش چهار-seed تیون‌شده، ensemble نماهای diffusion در Cora به ۸۸٫۶۵ درصد در مقابل ۸۸٫۸۸ برای DIVERGE و در CiteSeer به ۷۸٫۲۵ در مقابل ۷۸٫۲۱ می‌رسد. ترکیب دو مجموعه به ترتیب ۸۹٫۵۸ و ۷۹٫۰۸ است. اختلاف‌ها نسبت به پراکندگی seedها کوچک‌اند و شاهد برتری قطعی نیستند.

### ۵. افزایش دادهٔ نیمه‌نظارتی همراه با diffusion

`scripts/semi_aug_bridge.py` دادهٔ اصلی و دادهٔ افزوده را با پنج نمای LoRA و هشت نمای diffusion مقایسه می‌کند. چهار نسخهٔ diffusion روی PiSSA و joint روی چهار initialization دیگر ساخته می‌شوند. ensemble با میانگین احتمال انجام می‌شود.

`scripts/semi_methods_bridge.py` روش‌های ref_plain، ref_selected، com_vote، own_attn، own_viewdis، lit_conf، diva2_ens، diva_mi و diva_full را از مخزن خارجی DIVERGE فراخوانی می‌کند. تعریف داخلی این روش‌ها در همین مخزن نیست؛ از روی نام آنها نمی‌توان الگوریتم دقیق را نتیجه گرفت. نتایج ذخیره‌شدهٔ هر روش و تعداد تکرارهای واقعی در جدول‌های بعد آمده‌اند. تقسیم low-label این بخش با تقسیم ۶۰/۲۰/۲۰ پل قبلی متفاوت است و اعدادشان مقایسهٔ مستقیم ندارند.

### ۶. CF-DIVE — انتخاب فعال برچسب

کد `cf_dive` عدم‌قطعیت، اختلاف ensemble، تنوع معنایی و حساسیت پیش‌بینی به افزودن یک یال را ترکیب می‌کند تا نود مناسب برای دریافت برچسب انتخاب شود. نسخهٔ Efficient از دو نمایش MiniLM و TwinMask و طبقه‌بند Mean GraphSAGE استفاده می‌کند. بودجه از ۱۶ تا ۴۰ برچسب افزایش می‌یابد. کنترل matched no-CF همان روش با وزن حساسیت صفر است. انتخاب تصادفی، تک‌نما، ensemble خام و MILP نیز کنترل هستند. برای تابع پوشش فعلی، greedy و MILP به نتیجهٔ یکسان رسیده‌اند.

بهبود اولیه در تکرار ۲۰ seed تثبیت نشد: اختلاف CF با کنترل در Cora برابر −۰٫۰۴، در CiteSeer برابر +۰٫۵۹، در PubMed برابر −۰٫۷۴ و در WikiCS برابر −۰٫۷۸ واحد درصد AULC است. هر چهار بازهٔ اطمینان شامل صفر هستند. پس شواهد فعلی برای فعال‌سازی پیش‌فرض CF کافی نیست.

`fixed_labels.py` مقایسه با تعداد برچسب ثابت را پیاده می‌کند. `combined.py` و `diverge_views.py` ترکیب پنج نمای LoRA و هشت diffusion برای انتخاب فعال را آماده می‌کنند؛ وجود کد را نباید نتیجهٔ آزمایش نهایی دانست. در خروجی‌های بررسی‌شده، گزارش مستقل نهایی این ترکیب مشاهده نشد.

### ۷. TwinMask-Path و سیاست RL

`twinmask/denoise.py` بازسازی چندمرحله‌ای را روی checkpoint ثابت اجرا می‌کند. مسیرها عبارت‌اند از diagonal، diagonal_random، text_first، edge_first، text_lead1/2، edge_lead1/2 و joint_conf. مسیر ثابت بر اساس نودهای fit مجزا انتخاب می‌شود.

در `twinmask/path_rl.py` یک MLP سهم بودجهٔ متن را از مجموعهٔ صفر، ۰٫۲۵، ۰٫۵، ۰٫۷۵ و یک انتخاب می‌کند. سیاست با گرادیان گروهی شبیه GRPO آموزش می‌بیند. حالت teacher در گام آشکارسازی مقدار صحیح را برمی‌گرداند و یک کنترل تشخیصی است؛ حالت generate از پیش‌بینی مدل استفاده می‌کند. نتایج این دو حالت معادل نیستند.

در generate با rho=0.5، مسیر edge_lead2 در Cora، NLL متن را از 4.817 به 4.547 و در PubMed از 3.451 به 3.117 کاهش می‌دهد. سیاست RL در همین دو مورد به 4.794 و 3.403 می‌رسد و از این مسیر ثابت بهتر نیست. تولید با rho=1 در نمونه‌های Cora و CiteSeer، AUC نزدیک شانس دارد؛ بنابراین وجود نمونه‌سازی چندمرحله‌ای به معنی تولید موفق گراف کامل نیست. نتایج تمام مسیرها و تعداد گام‌ها در پیوست آمده‌اند.

## راهنمای معیارها و محدودیت‌ها

AUC و AP تشخیص لینک، MRR و Hits رتبه‌بندی، accuracy و Macro-F1 طبقه‌بندی و NLL بازسازی را می‌سنجند. NLL کمتر بهتر است؛ سایر معیارهای ذکرشده بیشتر بهترند. AULC مساحت نرمال‌شدهٔ منحنی دقت برحسب بودجه است و با دقت نهایی فرق دارد. HeaRT از منفی‌های سخت استفاده می‌کند و جایگزین AUC با منفی تصادفی نیست.

در جدول‌های استخراج‌شده، ± انحراف معیار روی اجراهای ذخیره‌شده است، نه بازهٔ اطمینان. فایل‌های `results/_old` آرشیوی هستند. برای استنتاج علمی باید تعداد seed، تقسیم داده، بودجه و معیار یکسان باشند. گزارش‌های پیوست متن منابع موجود هستند و ادعاهای ادبیات آنها در این کار مستقلاً بررسی نشده‌اند. لینک‌های نسبی داخل پیوست‌ها نسبت به پوشهٔ گزارش اصلی خودشان تعریف شده‌اند.

## جدول‌های خروجی و گزارش‌های تفصیلی


### `results/semi_aug_bridge/citeseer.json`

| / |  |  |     (%) |
|---|---|---:|---:|
| base | ENS DIVERGE-1 (5 inits) | 3 | 71.60  0.68 |
| base | ENS diffusion (8 views) | 3 | 71.55  0.28 |
| base | ENS DIVERGE-1 + diffusion | 3 | 71.93  0.19 |
| base | ENS screened all views | 3 | 71.65  0.81 |
| augFull | ENS DIVERGE-1 (5 inits) | 3 | 73.92  0.10 |
| augFull | ENS diffusion (8 views) | 3 | 73.68  0.02 |
| augFull | ENS DIVERGE-1 + diffusion | 3 | 74.08  0.14 |
| augFull | ENS screened all views | 3 | 73.50  0.19 |

### `results/semi_aug_bridge/cora.json`

| / |  |  |     (%) |
|---|---|---:|---:|
| base | ENS DIVERGE-1 (5 inits) | 3 | 84.48  0.51 |
| base | ENS diffusion (8 views) | 3 | 82.79  0.32 |
| base | ENS DIVERGE-1 + diffusion | 3 | 83.67  0.19 |
| base | ENS screened all views | 3 | 84.07  0.29 |
| augFull | ENS DIVERGE-1 (5 inits) | 3 | 84.75  0.44 |
| augFull | ENS diffusion (8 views) | 3 | 85.09  0.29 |
| augFull | ENS DIVERGE-1 + diffusion | 3 | 85.49  0.34 |
| augFull | ENS screened all views | 3 | 83.67  1.01 |

### `results/semi_aug_bridge/cora_naive_knn.json`

| / |  |  |     (%) |
|---|---|---:|---:|
| base | ENS DIVERGE-1 (5 inits) | 3 | 83.95  0.28 |
| base | ENS diffusion (8 views) | 3 | 82.54  0.49 |
| base | ENS DIVERGE-1 + diffusion | 3 | 83.51  0.36 |
| base | ENS screened all views | 3 | 83.28  0.30 |
| augFull | ENS DIVERGE-1 (5 inits) | 3 | 73.16  1.55 |
| augFull | ENS diffusion (8 views) | 3 | 77.80  0.20 |
| augFull | ENS DIVERGE-1 + diffusion | 3 | 77.80  0.83 |
| augFull | ENS screened all views | 3 | 74.97  1.34 |

### `results/semi_aug_bridge/methods_citeseer.json`

| / |  |  |     (%) |
|---|---|---:|---:|
| ref_plain | ENS DIVERGE-1 | 3 | 73.92  0.26 |
| ref_plain | ENS diffusion | 3 | 74.05  0.15 |
| ref_plain | ENS both | 3 | 74.19  0.24 |
| ref_selected | ENS DIVERGE-1 | 3 | 74.32  0.49 |
| ref_selected | ENS diffusion | 3 | 73.75  0.33 |
| ref_selected | ENS both | 3 | 73.94  0.16 |
| com_vote | ENS DIVERGE-1 | 3 | 73.69  0.46 |
| com_vote | ENS diffusion | 3 | 73.88  0.27 |
| com_vote | ENS both | 3 | 74.31  0.33 |
| own_attn | ENS DIVERGE-1 | 3 | 74.08  0.22 |
| own_attn | ENS diffusion | 3 | 74.28  0.35 |
| own_attn | ENS both | 3 | 74.25  0.29 |
| own_viewdis | ENS DIVERGE-1 | 3 | 74.06  0.08 |
| own_viewdis | ENS diffusion | 3 | 73.86  0.51 |
| own_viewdis | ENS both | 3 | 74.07  0.54 |
| lit_conf | ENS DIVERGE-1 | 3 | 74.18  0.10 |
| lit_conf | ENS diffusion | 3 | 73.89  0.36 |
| lit_conf | ENS both | 3 | 74.19  0.18 |
| diva2_ens | ENS DIVERGE-1 | 3 | 74.37  0.07 |
| diva2_ens | ENS diffusion | 3 | 74.20  0.25 |
| diva2_ens | ENS both | 3 | 74.24  0.25 |
| diva_mi | ENS DIVERGE-1 | 3 | 73.43  0.26 |
| diva_mi | ENS diffusion | 3 | 74.02  0.19 |
| diva_mi | ENS both | 3 | 74.21  0.13 |
| diva_full | ENS DIVERGE-1 | 3 | 72.80  0.17 |
| diva_full | ENS diffusion | 3 | 73.69  0.38 |
| diva_full | ENS both | 3 | 73.73  0.21 |

### `results/semi_aug_bridge/methods_cora.json`

| / |  |  |     (%) |
|---|---|---:|---:|
| ref_plain | ENS DIVERGE-1 | 3 | 85.15  0.65 |
| ref_plain | ENS diffusion | 3 | 84.77  0.58 |
| ref_plain | ENS both | 3 | 85.15  0.55 |
| ref_selected | ENS DIVERGE-1 | 3 | 85.15  0.28 |
| ref_selected | ENS diffusion | 3 | 84.93  0.49 |
| ref_selected | ENS both | 3 | 85.19  0.31 |
| com_vote | ENS DIVERGE-1 | 3 | 85.09  0.05 |
| com_vote | ENS diffusion | 3 | 85.06  0.36 |
| com_vote | ENS both | 3 | 85.43  0.13 |
| own_attn | ENS DIVERGE-1 | 3 | 85.38  0.39 |
| own_attn | ENS diffusion | 3 | 85.12  0.22 |
| own_attn | ENS both | 3 | 85.85  0.43 |
| own_viewdis | ENS DIVERGE-1 | 3 | 85.49  0.77 |
| own_viewdis | ENS diffusion | 3 | 85.06  0.12 |
| own_viewdis | ENS both | 3 | 85.54  0.64 |
| lit_conf | ENS DIVERGE-1 | 3 | 85.43  0.51 |
| lit_conf | ENS diffusion | 3 | 85.17  0.54 |
| lit_conf | ENS both | 3 | 85.41  0.54 |
| diva2_ens | ENS DIVERGE-1 | 3 | 84.95  0.58 |
| diva2_ens | ENS diffusion | 3 | 84.86  0.14 |
| diva2_ens | ENS both | 3 | 85.17  0.36 |
| diva_mi | ENS DIVERGE-1 | 3 | 83.82  0.38 |
| diva_mi | ENS diffusion | 3 | 84.11  0.16 |
| diva_mi | ENS both | 3 | 84.53  0.27 |
| diva_full | ENS DIVERGE-1 | 3 | 84.53  0.25 |
| diva_full | ENS diffusion | 3 | 84.78  0.61 |
| diva_full | ENS both | 3 | 85.14  0.55 |

### `results/diverge_bridge/citeseer.json`

| / |  |  |     (%) |
|---|---|---:|---:|
| all | ENS_diverge5 (5 LoRA inits) | 4 | 78.21  1.64 |
| all | ENS_minilm+4 diffusion views | 4 | 77.31  1.86 |
| all | ENS_minilm raw + 4 diffusion views | 4 | 77.43  1.89 |
| all | ENS_gemma_pissa+4 diffusion views | 4 | 77.63  1.78 |
| all | ENS_gemma_pissa raw + 4 diffusion views | 4 | 77.59  1.95 |
| all | ENS_diverge5 + gemma diffusion views | 4 | 78.06  2.02 |

### `results/diverge_bridge/cora.json`

| / |  |  |     (%) |
|---|---|---:|---:|
| all | ENS_diverge5 (5 LoRA inits) | 4 | 88.88  1.40 |
| all | ENS_minilm+4 diffusion views | 4 | 88.33  0.30 |
| all | ENS_minilm raw + 4 diffusion views | 4 | 88.65  0.62 |
| all | ENS_gemma_pissa+4 diffusion views | 4 | 88.33  1.38 |
| all | ENS_gemma_pissa raw + 4 diffusion views | 4 | 88.51  1.17 |
| all | ENS_diverge5 + gemma diffusion views | 4 | 88.47  1.21 |

### `results/diverge_bridge_tuned/citeseer.json`

| / |  |  |     (%) |
|---|---|---:|---:|
| all | ENS uniform  DIVERGE-5 | 4 | 78.21  1.64 |
| all | ENS learned  DIVERGE-5 | 4 | 78.10  2.03 |
| all | ENS uniform  MiniLM+4 diffusion (tuned) | 4 | 78.25  1.17 |
| all | ENS learned  MiniLM+4 diffusion (tuned) | 4 | 78.06  1.20 |
| all | ENS learned  DIVERGE-5 + our 5 views | 4 | 79.08  1.68 |

### `results/diverge_bridge_tuned/cora.json`

| / |  |  |     (%) |
|---|---|---:|---:|
| all | ENS uniform  DIVERGE-5 | 4 | 88.88  1.40 |
| all | ENS learned  DIVERGE-5 | 4 | 88.70  1.29 |
| all | ENS uniform  MiniLM+4 diffusion (tuned) | 4 | 88.65  0.71 |
| all | ENS learned  MiniLM+4 diffusion (tuned) | 4 | 88.47  0.66 |
| all | ENS learned  DIVERGE-5 + our 5 views | 4 | 89.58  0.55 |


---

## Detailed results appendix



# TwinMask — experimental results (auto-generated)

### Phase 0 — Link prediction (random negatives: AUC; HeaRT hard negatives: MRR, Hits@20), cold-start AUC, node classification (low-label public split / supervised 60-20-20); all in %, mean ± std over seeds

| dataset | variant | LP AUC | HeaRT MRR | HeaRT H@20 | cold AUC | NC low-label | NC 60/20/20 |
|---|---|---|---|---|---|---|---|
| cora | text_cosine | 92.77 ± 0.00 | 20.59 ± 0.00 | 49.47 ± 0.00 | 92.41 ± 0.00 | 72.68 ± 0.00 | 79.00 ± 0.00 |
| cora | gae | 96.85 ± 0.26 | 16.72 ± 0.19 | 55.23 ± 0.72 | 94.14 ± 0.20 | 73.82 ± 0.63 | 83.11 ± 0.48 |
| cora | mlp_lp | 96.48 ± 0.19 | 17.26 ± 0.20 | 57.19 ± 0.95 | 95.46 ± 0.32 | 76.48 ± 0.88 | 82.97 ± 0.26 |
| cora | edge_only | 97.18 ± 0.21 | 15.70 ± 1.30 | 55.16 ± 2.36 | 95.28 ± 0.14 | 77.03 ± 1.55 | 83.59 ± 0.58 |
| cora | text_only | – | – | – | – | 77.13 ± 0.55 | 85.18 ± 0.07 |
| cora | tied | 97.16 ± 0.28 | 13.74 ± 1.82 | 55.23 ± 2.16 | 95.24 ± 0.45 | 76.79 ± 0.69 | 84.59 ± 0.22 |
| cora | joint | 97.19 ± 0.05 | 14.50 ± 0.57 | 58.81 ± 0.36 | 95.10 ± 0.27 | 77.56 ± 1.06 | 84.92 ± 0.60 |
| citeseer | text_cosine | 96.07 ± 0.00 | 29.57 ± 0.00 | 61.07 ± 0.00 | 95.26 ± 0.00 | 68.67 ± 0.00 | 66.29 ± 0.00 |
| citeseer | gae | 97.95 ± 0.14 | 27.74 ± 0.55 | 69.69 ± 0.70 | 96.10 ± 0.22 | 63.60 ± 0.99 | 67.61 ± 0.17 |
| citeseer | mlp_lp | 98.24 ± 0.12 | 27.82 ± 0.49 | 70.31 ± 0.63 | 97.10 ± 0.20 | 67.29 ± 0.35 | 68.09 ± 0.28 |
| citeseer | edge_only | 98.30 ± 0.22 | 27.76 ± 1.27 | 70.93 ± 0.58 | 96.75 ± 0.04 | 67.17 ± 0.27 | 68.24 ± 0.38 |
| citeseer | text_only | – | – | – | – | 68.08 ± 0.33 | 69.78 ± 0.68 |
| citeseer | tied | 98.64 ± 0.04 | 24.92 ± 1.51 | 70.76 ± 0.82 | 97.21 ± 0.09 | 68.17 ± 0.13 | 71.59 ± 0.28 |
| citeseer | joint | 98.60 ± 0.06 | 24.18 ± 0.65 | 70.22 ± 1.45 | 97.10 ± 0.11 | 67.61 ± 0.46 | 70.39 ± 0.47 |
| pubmed | text_cosine | 95.10 ± 0.00 | 13.58 ± 0.00 | 35.70 ± 0.00 | 95.27 ± 0.00 | 75.75 ± 0.00 | 88.01 ± 0.00 |
| pubmed | gae | 98.87 ± 0.05 | 7.63 ± 0.50 | 36.13 ± 1.27 | 97.15 ± 0.09 | 79.92 ± 0.17 | 88.09 ± 0.12 |
| pubmed | mlp_lp | 97.92 ± 0.05 | 5.82 ± 0.32 | 27.83 ± 0.74 | 97.58 ± 0.07 | 80.18 ± 0.44 | 88.40 ± 0.09 |
| pubmed | edge_only | 99.06 ± 0.03 | 8.32 ± 0.31 | 37.37 ± 1.06 | 97.63 ± 0.05 | 80.54 ± 0.21 | 88.55 ± 0.09 |
| pubmed | text_only | – | – | – | – | 75.69 ± 1.62 | 88.33 ± 0.14 |
| pubmed | tied | 99.14 ± 0.08 | 8.80 ± 0.93 | 40.00 ± 2.98 | 97.64 ± 0.07 | 81.11 ± 0.44 | 88.95 ± 0.09 |
| pubmed | joint | 99.22 ± 0.02 | 7.66 ± 0.52 | 38.43 ± 1.27 | 97.62 ± 0.13 | 81.82 ± 0.65 | 89.06 ± 0.12 |
| wikics | text_cosine | 88.58 ± 0.00 | 8.14 ± 0.00 | 19.30 ± 0.00 | 87.43 ± 0.00 | 69.83 ± 0.00 | 73.43 ± 0.00 |
| wikics | gae | 99.22 ± 0.01 | 23.01 ± 0.90 | 55.13 ± 1.42 | 95.38 ± 0.42 | 72.05 ± 0.63 | 78.75 ± 0.03 |
| wikics | mlp_lp | 96.08 ± 0.42 | 10.34 ± 3.72 | 34.83 ± 7.21 | 95.45 ± 0.74 | 74.91 ± 0.21 | 78.79 ± 0.38 |
| wikics | edge_only | 99.25 ± 0.01 | 18.72 ± 2.69 | 52.23 ± 0.66 | 96.89 ± 0.07 | 73.37 ± 0.31 | 78.89 ± 0.16 |
| wikics | text_only | – | – | – | – | 77.33 ± 0.38 | 80.94 ± 0.28 |
| wikics | tied | 99.15 ± 0.02 | 15.58 ± 0.27 | 46.93 ± 0.34 | 96.58 ± 0.06 | 75.81 ± 0.24 | 81.09 ± 0.19 |
| wikics | joint | 99.14 ± 0.00 | 19.10 ± 1.06 | 52.23 ± 0.76 | 96.91 ± 0.05 | 76.92 ± 0.31 | 81.14 ± 0.32 |
| instagram | text_cosine | 73.70 ± 0.00 | 2.99 ± 0.00 | 6.20 ± 0.00 | 71.18 ± 0.00 | 58.17 ± 0.00 | 64.81 ± 0.00 |
| instagram | gae | 96.33 ± 0.04 | 7.64 ± 0.46 | 31.87 ± 0.57 | 84.32 ± 0.28 | 61.06 ± 0.33 | 66.24 ± 0.27 |
| instagram | mlp_lp | 94.31 ± 0.07 | 3.23 ± 0.05 | 12.83 ± 0.40 | 84.29 ± 0.37 | 60.81 ± 0.57 | 66.31 ± 0.38 |
| instagram | edge_only | 96.80 ± 0.07 | 6.15 ± 0.73 | 26.20 ± 2.41 | 86.04 ± 0.13 | 61.43 ± 0.32 | 66.44 ± 0.07 |
| instagram | text_only | – | – | – | – | 60.86 ± 0.56 | 66.61 ± 0.09 |
| instagram | tied | 96.95 ± 0.04 | 7.14 ± 0.34 | 29.63 ± 0.58 | 86.70 ± 0.16 | 61.61 ± 0.12 | 66.49 ± 0.12 |
| instagram | joint | 97.00 ± 0.02 | 6.70 ± 0.48 | 29.33 ± 1.02 | 87.32 ± 0.17 | 61.37 ± 0.59 | 66.67 ± 0.22 |
| reddit | text_cosine | 69.92 ± 0.00 | 3.09 ± 0.00 | 6.50 ± 0.00 | 70.22 ± 0.00 | 56.86 ± 0.00 | 60.18 ± 0.00 |
| reddit | gae | 88.30 ± 0.02 | 3.29 ± 0.12 | 13.20 ± 0.33 | 80.15 ± 0.36 | 65.82 ± 0.14 | 68.21 ± 0.27 |
| reddit | mlp_lp | 86.83 ± 0.13 | 3.16 ± 0.10 | 10.20 ± 0.43 | 80.73 ± 0.38 | 64.33 ± 0.78 | 66.63 ± 0.42 |
| reddit | edge_only | 90.74 ± 0.07 | 3.44 ± 0.11 | 12.30 ± 0.16 | 83.47 ± 0.17 | 65.72 ± 0.07 | 68.01 ± 0.03 |
| reddit | text_only | – | – | – | – | 65.39 ± 0.14 | 67.59 ± 0.20 |
| reddit | tied | 91.57 ± 0.02 | 3.72 ± 0.07 | 15.80 ± 0.22 | 83.87 ± 0.17 | 66.10 ± 0.13 | 68.24 ± 0.11 |
| reddit | joint | 91.84 ± 0.07 | 3.89 ± 0.12 | 16.57 ± 0.95 | 83.86 ± 0.14 | 66.21 ± 0.08 | 68.24 ± 0.16 |

### Phase 0 — Text-embedding denoising: cosine(x̂0, x0) of test nodes, graph visible vs hidden

| dataset | variant | t | noisy input | no graph | with graph | Δ (graph − no graph) |
|---|---|---|---|---|---|---|
| cora | text_only | 0.5 | 0.763 | 0.830 | 0.848 | +0.018 |
| cora | text_only | 0.9 | 0.315 | 0.496 | 0.642 | +0.146 |
| cora | joint | 0.5 | 0.763 | 0.838 | 0.841 | +0.003 |
| cora | joint | 0.9 | 0.315 | 0.512 | 0.601 | +0.089 |
| cora | tied | 0.5 | 0.763 | 0.827 | 0.829 | +0.002 |
| cora | tied | 0.9 | 0.315 | 0.505 | 0.544 | +0.039 |
| citeseer | text_only | 0.5 | 0.760 | 0.845 | 0.853 | +0.008 |
| citeseer | text_only | 0.9 | 0.307 | 0.522 | 0.638 | +0.116 |
| citeseer | joint | 0.5 | 0.760 | 0.841 | 0.842 | +0.001 |
| citeseer | joint | 0.9 | 0.307 | 0.526 | 0.607 | +0.081 |
| citeseer | tied | 0.5 | 0.760 | 0.832 | 0.833 | +0.001 |
| citeseer | tied | 0.9 | 0.307 | 0.511 | 0.556 | +0.045 |
| pubmed | text_only | 0.5 | 0.812 | 0.895 | 0.895 | +0.000 |
| pubmed | text_only | 0.9 | 0.457 | 0.665 | 0.704 | +0.039 |
| pubmed | joint | 0.5 | 0.812 | 0.882 | 0.883 | +0.000 |
| pubmed | joint | 0.9 | 0.457 | 0.664 | 0.697 | +0.032 |
| pubmed | tied | 0.5 | 0.812 | 0.870 | 0.870 | +0.001 |
| pubmed | tied | 0.9 | 0.457 | 0.653 | 0.669 | +0.016 |
| wikics | text_only | 0.5 | 0.769 | 0.840 | 0.848 | +0.008 |
| wikics | text_only | 0.9 | 0.334 | 0.508 | 0.647 | +0.139 |
| wikics | joint | 0.5 | 0.769 | 0.835 | 0.839 | +0.004 |
| wikics | joint | 0.9 | 0.334 | 0.509 | 0.627 | +0.118 |
| wikics | tied | 0.5 | 0.769 | 0.824 | 0.827 | +0.003 |
| wikics | tied | 0.9 | 0.334 | 0.509 | 0.559 | +0.051 |
| instagram | text_only | 0.5 | 0.775 | 0.858 | 0.859 | +0.001 |
| instagram | text_only | 0.9 | 0.352 | 0.543 | 0.566 | +0.023 |
| instagram | joint | 0.5 | 0.775 | 0.843 | 0.845 | +0.001 |
| instagram | joint | 0.9 | 0.352 | 0.539 | 0.560 | +0.021 |
| instagram | tied | 0.5 | 0.775 | 0.833 | 0.835 | +0.002 |
| instagram | tied | 0.9 | 0.352 | 0.531 | 0.542 | +0.010 |
| reddit | text_only | 0.5 | 0.724 | 0.807 | 0.807 | -0.001 |
| reddit | text_only | 0.9 | 0.204 | 0.326 | 0.374 | +0.047 |
| reddit | joint | 0.5 | 0.724 | 0.794 | 0.795 | +0.001 |
| reddit | joint | 0.9 | 0.204 | 0.323 | 0.365 | +0.042 |
| reddit | tied | 0.5 | 0.724 | 0.782 | 0.783 | +0.001 |
| reddit | tied | 0.9 | 0.204 | 0.311 | 0.327 | +0.016 |

### Phase 1/2 (TwinMask-S, DistilRoBERTa backbone) — link prediction / cold-start / node classification

| dataset | variant | seed | LP AUC (full) | LP AUC text masked | LP AUC pair only | HeaRT MRR | HeaRT H@20 | cold AUC | NC low-label | NC 60/20/20 |
|---|---|---|---|---|---|---|---|---|---|---|
| citeseer | edge_only | 0 | 89.18 | 45.64 | 88.56 | 24.69 | 60.80 | 89.59 | 61.15 | 64.92 |
| citeseer | joint | 0 | 89.98 | 57.44 | 87.90 | 16.26 | 53.60 | 89.53 | 67.26 | 67.48 |
| citeseer | text_only | 0 | – | – | – | – | – | – | 65.78 | 67.07 |
| citeseer | tied | 0 | 90.23 | 61.30 | 86.80 | 15.02 | 57.60 | 85.02 | 64.61 | 67.12 |
| cora | edge_only | 0 | 84.98 | 42.47 | 84.52 | 15.97 | 47.79 | 83.64 | 59.91 | 73.54 |
| cora | joint | 0 | 89.79 | 64.51 | 89.28 | 14.69 | 47.16 | 82.97 | 69.87 | 79.89 |
| cora | joint | 1 | 89.39 | 63.35 | 87.99 | 13.18 | 44.63 | 82.00 | 70.55 | 79.11 |
| cora | notopo | 0 | 86.81 | 61.33 | 86.47 | 13.27 | 45.47 | 84.57 | 69.15 | 79.67 |
| cora | static | 0 | 89.76 | 63.44 | 88.33 | 14.81 | 46.95 | 85.36 | 70.84 | 79.58 |
| cora | text_only | 0 | – | – | – | – | – | – | 70.60 | 80.90 |
| cora | tied | 0 | 89.54 | 67.02 | 84.60 | 15.82 | 52.63 | 81.47 | 73.02 | 80.42 |
| instagram | joint | 0 | 90.72 | 85.28 | 77.55 | 8.78 | 26.30 | 77.22 | 58.47 | 64.18 |
| pubmed | edge_only | 0 | 97.25 | 78.44 | 96.65 | 24.60 | 58.60 | 90.64 | 75.67 | 88.33 |
| pubmed | joint | 0 | 96.51 | 86.87 | 93.82 | 18.36 | 49.20 | 89.40 | 76.35 | 88.09 |
| pubmed | notopo | 0 | 95.63 | 79.73 | 93.58 | 20.40 | 55.50 | 90.31 | 75.36 | 88.05 |
| pubmed | static | 0 | 96.37 | 81.91 | 93.73 | 22.69 | 56.20 | 90.61 | 76.66 | 87.96 |
| pubmed | text_only | 0 | – | – | – | – | – | – | 76.66 | 87.39 |
| pubmed | tied | 0 | 96.62 | 88.72 | 90.54 | 20.83 | 52.10 | 87.58 | 75.96 | 87.67 |
| reddit | joint | 0 | 77.58 | 69.92 | 74.39 | 3.48 | 13.00 | 67.53 | 63.95 | 67.73 |
| wikics | joint | 0 | 97.47 | 95.12 | 93.50 | 25.34 | 54.90 | 94.95 | 77.49 | 79.63 |

### Phase 1/2 (TwinMask-S, DistilRoBERTa backbone) — text completion of a test node (masked-token NLL ↓ / top-1 acc ↑)

| dataset | variant | mask ratio | alone | subgraph, edges [M] | subgraph + edges | NLL gain of context |
|---|---|---|---|---|---|---|
| citeseer | joint | 0.5 | 4.741 / 24.2% | 4.611 / 25.5% | 4.615 / 25.4% | +0.126 |
| citeseer | joint | 1.0 | 7.137 / 4.8% | 7.045 / 4.9% | 7.039 / 4.9% | +0.099 |
| citeseer | text_only | 0.5 | 3.825 / 32.8% | 3.563 / 35.4% | 3.459 / 36.8% | +0.366 |
| citeseer | text_only | 1.0 | 6.912 / 5.2% | 6.636 / 5.3% | 6.510 / 5.5% | +0.402 |
| citeseer | tied | 0.5 | 4.174 / 29.7% | 4.032 / 30.9% | 4.024 / 31.1% | +0.150 |
| citeseer | tied | 1.0 | 6.974 / 5.1% | 6.866 / 5.0% | 6.840 / 5.0% | +0.135 |
| cora | joint | 0.5 | 4.192 / 29.7% | 4.112 / 30.6% | 4.091 / 30.7% | +0.101 |
| cora | joint | 1.0 | 6.954 / 5.9% | 6.819 / 5.8% | 6.816 / 5.9% | +0.138 |
| cora | joint | 0.5 | 4.612 / 25.8% | 4.553 / 26.0% | 4.550 / 26.0% | +0.062 |
| cora | joint | 1.0 | 7.111 / 5.5% | 7.013 / 5.7% | 7.006 / 5.8% | +0.105 |
| cora | notopo | 0.5 | 4.193 / 29.8% | 4.108 / 30.8% | 4.101 / 30.7% | +0.091 |
| cora | notopo | 1.0 | 6.964 / 6.0% | 6.859 / 5.9% | 6.894 / 5.8% | +0.070 |
| cora | static | 0.5 | 4.180 / 29.8% | 4.101 / 30.6% | 4.087 / 30.6% | +0.093 |
| cora | static | 1.0 | 6.959 / 6.2% | 6.845 / 6.1% | 6.842 / 6.1% | +0.117 |
| cora | text_only | 0.5 | 3.441 / 37.0% | 3.254 / 38.7% | 3.143 / 39.9% | +0.298 |
| cora | text_only | 1.0 | 6.849 / 5.9% | 6.567 / 6.3% | 6.400 / 6.2% | +0.449 |
| cora | tied | 0.5 | 3.918 / 32.3% | 3.827 / 33.2% | 3.817 / 33.4% | +0.102 |
| cora | tied | 1.0 | 6.910 / 6.2% | 6.759 / 6.1% | 6.741 / 6.1% | +0.168 |
| instagram | joint | 0.5 | 4.476 / 30.1% | 4.402 / 30.8% | 4.438 / 30.9% | +0.038 |
| instagram | joint | 1.0 | 6.956 / 5.8% | 6.921 / 5.9% | 6.914 / 6.1% | +0.042 |
| pubmed | joint | 0.5 | 3.125 / 45.6% | 3.086 / 46.1% | 3.094 / 46.0% | +0.031 |
| pubmed | joint | 1.0 | 6.457 / 9.2% | 6.260 / 9.0% | 6.270 / 9.1% | +0.187 |
| pubmed | notopo | 0.5 | 3.142 / 45.3% | 3.096 / 46.1% | 3.103 / 46.1% | +0.039 |
| pubmed | notopo | 1.0 | 6.458 / 8.9% | 6.278 / 9.0% | 6.296 / 8.9% | +0.162 |
| pubmed | static | 0.5 | 3.125 / 45.5% | 3.091 / 46.2% | 3.099 / 45.9% | +0.026 |
| pubmed | static | 1.0 | 6.455 / 9.1% | 6.276 / 9.1% | 6.272 / 9.0% | +0.183 |
| pubmed | text_only | 0.5 | 2.935 / 48.0% | 2.890 / 48.5% | 2.881 / 48.6% | +0.055 |
| pubmed | text_only | 1.0 | 6.415 / 9.2% | 6.198 / 9.1% | 6.174 / 9.0% | +0.241 |
| pubmed | tied | 0.5 | 2.976 / 47.3% | 2.925 / 48.3% | 2.929 / 48.2% | +0.047 |
| pubmed | tied | 1.0 | 6.429 / 9.2% | 6.204 / 9.2% | 6.205 / 9.2% | +0.224 |
| reddit | joint | 0.5 | 4.522 / 25.2% | 4.472 / 26.0% | 4.481 / 26.2% | +0.041 |
| reddit | joint | 1.0 | 6.956 / 5.4% | 6.930 / 5.5% | 6.942 / 5.5% | +0.014 |
| wikics | joint | 0.5 | 5.105 / 25.0% | 4.997 / 25.9% | 5.018 / 25.6% | +0.086 |
| wikics | joint | 1.0 | 7.868 / 5.6% | 7.729 / 5.7% | 7.722 / 5.7% | +0.145 |

### Phase 2 — TSMI estimates (nats per item; NELBO differences, higher = stronger text↔structure dependence)

| dataset | homophily | TSMI_A (text→edges) | TSMI_X topology (edges→text) | CTX_X (neighbourhood→text) |
|---|---|---|---|---|
| cora | 0.810 | 0.1386 | 0.0226 | 0.0655 |
| pubmed | 0.802 | 0.0425 | -0.0020 | 0.0218 |
| citeseer | 0.744 | 0.1032 | 0.0076 | 0.0170 |
| wikics | 0.654 | 0.1307 | -0.0032 | -0.0094 |
| reddit | 0.588 | 0.0204 | -0.0013 | -0.0204 |
| instagram | 0.586 | 0.0074 | -0.0036 | -0.0241 |

![fig_heatmap_citeseer.png](fig_heatmap_citeseer.png)

![fig_heatmap_cora.png](fig_heatmap_cora.png)

![fig_heatmap_instagram.png](fig_heatmap_instagram.png)

![fig_heatmap_pubmed.png](fig_heatmap_pubmed.png)

![fig_heatmap_reddit.png](fig_heatmap_reddit.png)

![fig_heatmap_wikics.png](fig_heatmap_wikics.png)

![fig_tsmi_vs_homophily.png](fig_tsmi_vs_homophily.png)

### Phase 3 (ogbn-arxiv) — link prediction / cold-start / node classification

| dataset | variant | seed | LP AUC (full) | LP AUC text masked | LP AUC pair only | HeaRT MRR | HeaRT H@20 | cold AUC | NC low-label | NC 60/20/20 |
|---|---|---|---|---|---|---|---|---|---|---|
| arxiv | edge_only | 0 | 97.91 | 73.15 | 96.74 | 7.89 | 25.10 | 90.92 | 61.49 | 63.14 |
| arxiv | joint | 0 | 98.08 | 88.40 | 97.25 | 7.51 | 24.80 | 90.48 | 62.34 | 65.91 |

### Phase 3 (ogbn-arxiv) — text completion of a test node (masked-token NLL ↓ / top-1 acc ↑)

| dataset | variant | mask ratio | alone | subgraph, edges [M] | subgraph + edges | NLL gain of context |
|---|---|---|---|---|---|---|
| arxiv | joint | 0.5 | 3.762 / 35.9% | 3.655 / 37.2% | 3.652 / 37.3% | +0.111 |
| arxiv | joint | 1.0 | 7.050 / 5.9% | 6.759 / 6.0% | 6.771 / 6.0% | +0.279 |


---

## Detailed results appendix



# CF-DIVE experiment results

Each dataset uses one fixed transductive split. The paired intervals describe acquisition and model seed variability on that split. Historical pretraining, cache creation, process startup and result serialization are excluded from the timing.

## citeseer

Positive-CF validation choice γ=0.3, coverage bonus=0.05. Best unrestricted validation pair [0.0, 0.15]. Best zero-CF bonus 0.15.

Validation AULC in percent, mean ± sample SD over three seeds.

| γ / bonus | .05 | .15 | .30 |
| --- | --- | --- | --- |
| 0.0 | 64.42 ± 1.36 | 64.82 ± 1.07 | 64.82 ± 1.07 |
| 0.1 | 61.32 ± 4.47 | 62.28 ± 5.18 | 62.28 ± 5.18 |
| 0.3 | 64.33 ± 3.82 | 63.54 ± 4.30 | 63.54 ± 4.30 |
| 0.6 | 63.88 ± 4.87 | 62.86 ± 4.82 | 63.28 ± 4.39 |

Test accuracy by acquired labels, normalized accuracy AULC, and final macro-F1, all in percent.

| Method | 16 | 24 | 32 | 40 | AULC ± SD | F1 @ 40 |
| --- | --- | --- | --- | --- | --- | --- |
| random | 53.45 | 60.82 | 60.41 | 64.67 | 60.09 ± 4.75 | 58.29 |
| entropy | 53.45 | 59.34 | 61.76 | 66.14 | 60.30 ± 5.57 | 60.49 |
| disagreement | 53.45 | 60.66 | 61.72 | 65.64 | 60.64 ± 4.71 | 60.11 |
| no_cf | 53.45 | 60.69 | 64.39 | 67.30 | 61.82 ± 4.80 | 61.05 |
| tuned_no_cf | 53.45 | 60.69 | 64.36 | 67.34 | 61.81 ± 4.79 | 61.09 |
| cf | 53.45 | 63.57 | 66.55 | 67.96 | 63.61 ± 2.73 | 62.29 |
| cf_top | 53.45 | 62.70 | 65.05 | 65.77 | 62.45 ± 3.75 | 59.24 |
| cf_milp | 53.45 | 63.57 | 66.55 | 67.96 | 63.61 ± 2.73 | 62.29 |
| raw_ensemble_cf | 56.90 | 62.88 | 67.15 | 67.18 | 64.02 ± 2.62 | 61.45 |
| single_cf | 55.61 | 62.10 | 65.77 | 67.15 | 63.08 ± 2.32 | 61.70 |

Paired CF minus no_cf AULC in seed order 41–45 (percentage points): +4.00, +4.39, -0.63, -1.54, +2.74. Mean +1.79, descriptive paired t interval [-1.58, +5.16].

Across 2,055 independent edge probes in the CF arm, the average member flipped at the target in 16.84% and at the other endpoint in 9.25%.

Mean cumulative wall seconds by label budget on one device. Divide by 3,600 for allocated device-hours.

| Method | 16 | 24 | 32 | 40 |
| --- | --- | --- | --- | --- |
| random | 0.029 | 0.059 | 0.090 | 0.120 |
| entropy | 0.029 | 0.059 | 0.090 | 0.120 |
| disagreement | 0.028 | 0.059 | 0.089 | 0.120 |
| no_cf | 0.028 | 0.062 | 0.093 | 0.124 |
| tuned_no_cf | 0.029 | 0.060 | 0.091 | 0.122 |
| cf | 0.029 | 0.063 | 0.096 | 0.129 |
| cf_top | 0.029 | 0.064 | 0.097 | 0.129 |
| cf_milp | 0.028 | 0.064 | 0.100 | 0.136 |
| raw_ensemble_cf | 0.028 | 0.061 | 0.094 | 0.126 |
| single_cf | 0.015 | 0.032 | 0.049 | 0.066 |

Summed validation and test trial time 0.0029 device-hours. One-time preparation 0.19 seconds. Device: NVIDIA GeForce RTX 5070.

## cora

Positive-CF validation choice γ=0.3, coverage bonus=0.05. Best unrestricted validation pair [0.3, 0.05]. Best zero-CF bonus 0.05.

Validation AULC in percent, mean ± sample SD over three seeds.

| γ / bonus | .05 | .15 | .30 |
| --- | --- | --- | --- |
| 0.0 | 65.74 ± 3.84 | 65.74 ± 3.84 | 65.74 ± 3.84 |
| 0.1 | 67.58 ± 4.96 | 67.63 ± 4.89 | 67.63 ± 4.89 |
| 0.3 | 68.18 ± 3.88 | 67.79 ± 3.65 | 67.84 ± 3.54 |
| 0.6 | 66.20 ± 2.30 | 65.96 ± 1.79 | 67.79 ± 3.65 |

Test accuracy by acquired labels, normalized accuracy AULC, and final macro-F1, all in percent.

| Method | 16 | 24 | 32 | 40 | AULC ± SD | F1 @ 40 |
| --- | --- | --- | --- | --- | --- | --- |
| random | 63.87 | 67.64 | 70.96 | 74.24 | 69.22 ± 3.42 | 67.97 |
| entropy | 63.87 | 68.45 | 71.33 | 73.36 | 69.46 ± 1.43 | 69.67 |
| disagreement | 63.87 | 65.57 | 69.23 | 70.11 | 67.26 ± 3.63 | 63.06 |
| no_cf | 63.87 | 65.42 | 69.89 | 73.43 | 67.99 ± 1.74 | 69.13 |
| tuned_no_cf | 63.87 | 65.42 | 69.89 | 73.43 | 67.99 ± 1.74 | 69.13 |
| cf | 63.87 | 67.82 | 73.47 | 74.54 | 70.17 ± 1.81 | 70.89 |
| cf_top | 63.87 | 68.12 | 72.69 | 75.02 | 70.09 ± 1.66 | 71.11 |
| cf_milp | 63.87 | 67.82 | 73.47 | 74.54 | 70.17 ± 1.81 | 70.89 |
| raw_ensemble_cf | 68.52 | 73.06 | 75.57 | 76.35 | 73.69 ± 2.71 | 71.42 |
| single_cf | 68.19 | 71.55 | 73.36 | 77.23 | 72.54 ± 2.20 | 72.46 |

Paired CF minus no_cf AULC in seed order 41–45 (percentage points): +0.43, +2.12, +6.40, +1.72, +0.22. Mean +2.18, descriptive paired t interval [-0.92, +5.28].

Across 3,423 independent edge probes in the CF arm, the average member flipped at the target in 16.90% and at the other endpoint in 7.58%.

Mean cumulative wall seconds by label budget on one device. Divide by 3,600 for allocated device-hours.

| Method | 16 | 24 | 32 | 40 |
| --- | --- | --- | --- | --- |
| random | 0.028 | 0.058 | 0.088 | 0.117 |
| entropy | 0.029 | 0.059 | 0.090 | 0.121 |
| disagreement | 0.029 | 0.059 | 0.089 | 0.119 |
| no_cf | 0.029 | 0.060 | 0.090 | 0.121 |
| tuned_no_cf | 0.029 | 0.060 | 0.090 | 0.121 |
| cf | 0.028 | 0.061 | 0.093 | 0.126 |
| cf_top | 0.029 | 0.063 | 0.096 | 0.128 |
| cf_milp | 0.028 | 0.063 | 0.098 | 0.134 |
| raw_ensemble_cf | 0.028 | 0.060 | 0.092 | 0.124 |
| single_cf | 0.015 | 0.032 | 0.050 | 0.067 |

Summed validation and test trial time 0.0035 device-hours. One-time preparation 0.38 seconds. Device: NVIDIA GeForce RTX 5070.

## pubmed

Positive-CF validation choice γ=0.1, coverage bonus=0.05. Best unrestricted validation pair [0.1, 0.05]. Best zero-CF bonus 0.05.

Validation AULC in percent, mean ± sample SD over three seeds.

| γ / bonus | .05 | .15 | .30 |
| --- | --- | --- | --- |
| 0.0 | 72.94 ± 4.35 | 72.82 ± 4.56 | 72.82 ± 4.56 |
| 0.1 | 74.62 ± 2.57 | 74.27 ± 2.98 | 74.27 ± 2.98 |
| 0.3 | 72.72 ± 4.27 | 72.39 ± 4.31 | 72.39 ± 4.31 |
| 0.6 | 72.69 ± 4.24 | 72.81 ± 4.24 | 72.55 ± 4.23 |

Test accuracy by acquired labels, normalized accuracy AULC, and final macro-F1, all in percent.

| Method | 16 | 24 | 32 | 40 | AULC ± SD | F1 @ 40 |
| --- | --- | --- | --- | --- | --- | --- |
| random | 64.41 | 73.65 | 75.90 | 76.85 | 73.39 ± 2.18 | 76.62 |
| entropy | 64.41 | 73.92 | 77.17 | 78.27 | 74.14 ± 2.33 | 78.10 |
| disagreement | 64.41 | 68.74 | 71.64 | 72.20 | 69.56 ± 5.40 | 71.57 |
| no_cf | 64.41 | 72.77 | 72.61 | 74.86 | 71.67 ± 2.78 | 74.61 |
| tuned_no_cf | 64.41 | 72.77 | 72.61 | 74.86 | 71.67 ± 2.78 | 74.61 |
| cf | 64.41 | 72.59 | 73.66 | 75.44 | 72.05 ± 3.67 | 75.13 |
| cf_top | 64.41 | 71.28 | 71.40 | 73.76 | 70.59 ± 3.87 | 73.75 |
| cf_milp | 64.41 | 72.59 | 73.66 | 75.44 | 72.05 ± 3.67 | 75.13 |
| raw_ensemble_cf | 65.99 | 70.47 | 75.72 | 77.09 | 72.58 ± 1.17 | 76.84 |
| single_cf | 65.27 | 71.64 | 75.02 | 76.63 | 72.54 ± 1.19 | 76.43 |

Paired CF minus no_cf AULC in seed order 41–45 (percentage points): +1.27, -1.21, -2.18, +3.19, +0.85. Mean +0.38, descriptive paired t interval [-2.25, +3.02].

Across 3,811 independent edge probes in the CF arm, the average member flipped at the target in 15.36% and at the other endpoint in 9.60%.

Mean cumulative wall seconds by label budget on one device. Divide by 3,600 for allocated device-hours.

| Method | 16 | 24 | 32 | 40 |
| --- | --- | --- | --- | --- |
| random | 0.033 | 0.068 | 0.103 | 0.138 |
| entropy | 0.032 | 0.067 | 0.104 | 0.140 |
| disagreement | 0.035 | 0.072 | 0.108 | 0.145 |
| no_cf | 0.034 | 0.074 | 0.110 | 0.148 |
| tuned_no_cf | 0.034 | 0.071 | 0.108 | 0.148 |
| cf | 0.037 | 0.077 | 0.117 | 0.161 |
| cf_top | 0.035 | 0.075 | 0.117 | 0.159 |
| cf_milp | 0.036 | 0.079 | 0.121 | 0.166 |
| raw_ensemble_cf | 0.033 | 0.075 | 0.117 | 0.160 |
| single_cf | 0.017 | 0.040 | 0.062 | 0.084 |

Summed validation and test trial time 0.0037 device-hours. One-time preparation 1.43 seconds. Device: NVIDIA GeForce RTX 5070.

## wikics

Positive-CF validation choice γ=0.3, coverage bonus=0.05. Best unrestricted validation pair [0.0, 0.05]. Best zero-CF bonus 0.05.

Validation AULC in percent, mean ± sample SD over three seeds.

| γ / bonus | .05 | .15 | .30 |
| --- | --- | --- | --- |
| 0.0 | 58.39 ± 2.49 | 58.39 ± 2.49 | 58.39 ± 2.49 |
| 0.1 | 56.98 ± 7.82 | 56.97 ± 7.82 | 56.97 ± 7.82 |
| 0.3 | 57.84 ± 5.51 | 57.34 ± 7.40 | 57.38 ± 7.47 |
| 0.6 | 57.84 ± 5.51 | 57.84 ± 5.51 | 57.34 ± 7.40 |

Test accuracy by acquired labels, normalized accuracy AULC, and final macro-F1, all in percent.

| Method | 16 | 24 | 32 | 40 | AULC ± SD | F1 @ 40 |
| --- | --- | --- | --- | --- | --- | --- |
| random | 57.04 | 61.77 | 64.79 | 66.89 | 62.84 ± 4.26 | 54.87 |
| entropy | 57.04 | 60.19 | 61.32 | 65.81 | 60.98 ± 2.87 | 53.34 |
| disagreement | 57.04 | 60.49 | 62.85 | 65.36 | 61.51 ± 4.82 | 52.18 |
| no_cf | 57.04 | 60.55 | 64.80 | 66.81 | 62.42 ± 3.63 | 57.67 |
| tuned_no_cf | 57.04 | 60.55 | 64.80 | 66.81 | 62.42 ± 3.63 | 57.67 |
| cf | 57.04 | 62.01 | 64.74 | 66.38 | 62.82 ± 2.57 | 57.45 |
| cf_top | 57.04 | 59.05 | 63.40 | 65.47 | 61.24 ± 3.30 | 55.55 |
| cf_milp | 57.04 | 62.01 | 64.74 | 66.38 | 62.82 ± 2.57 | 57.45 |
| raw_ensemble_cf | 53.41 | 55.89 | 59.62 | 61.94 | 57.73 ± 3.88 | 52.52 |
| single_cf | 52.74 | 56.02 | 59.72 | 61.74 | 57.66 ± 2.77 | 54.31 |

Paired CF minus no_cf AULC in seed order 41–45 (percentage points): -0.67, -0.19, -0.90, -1.94, +5.67. Mean +0.40, descriptive paired t interval [-3.35, +4.15].

Across 3,126 independent edge probes in the CF arm, the average member flipped at the target in 7.98% and at the other endpoint in 1.97%.

Mean cumulative wall seconds by label budget on one device. Divide by 3,600 for allocated device-hours.

| Method | 16 | 24 | 32 | 40 |
| --- | --- | --- | --- | --- |
| random | 0.031 | 0.065 | 0.098 | 0.130 |
| entropy | 0.030 | 0.064 | 0.097 | 0.130 |
| disagreement | 0.030 | 0.064 | 0.097 | 0.129 |
| no_cf | 0.029 | 0.062 | 0.094 | 0.126 |
| tuned_no_cf | 0.033 | 0.071 | 0.111 | 0.152 |
| cf | 0.037 | 0.083 | 0.127 | 0.170 |
| cf_top | 0.036 | 0.082 | 0.125 | 0.169 |
| cf_milp | 0.033 | 0.073 | 0.115 | 0.156 |
| raw_ensemble_cf | 0.029 | 0.064 | 0.097 | 0.131 |
| single_cf | 0.015 | 0.033 | 0.052 | 0.070 |

Summed validation and test trial time 0.0032 device-hours. One-time preparation 1.26 seconds. Device: NVIDIA GeForce RTX 5070.


---

## Detailed results appendix



# Locked-configuration replication

Exploratory follow-up after inspecting seeds 41–45. Seeds 51–70 vary initial labels and model initialization on the original split. Values are normalized accuracy AULC in percent. Differences and unadjusted paired 95% t intervals are in percentage points.

| Dataset | CF | Matched no CF | Random | Paired Δ | 95% interval | CF wins / 20 |
| --- | --- | --- | --- | --- | --- | --- |
| citeseer | 66.31 | 65.72 | 65.60 | +0.59 | [-0.15, +1.33] | 13 |
| cora | 66.72 | 66.76 | 65.89 | -0.04 | [-0.97, +0.90] | 8 |
| pubmed | 71.51 | 72.25 | 72.54 | -0.74 | [-1.68, +0.21] | 6 |
| wikics | 59.74 | 60.52 | 61.12 | -0.78 | [-1.76, +0.21] | 8 |

![Label and time efficiency over 20 seeds](curves.png)


---

## Detailed results appendix



# TwinMask-Path results

Δ = paired mean difference to the diagonal path on the same episodes (95% CI); negative NLL Δ = better.

### citeseer (homophily 0.74) — rho=0.5, N=8, mode=generate, test cold nodes=79 x 3 resamplings
tuned fixed path (selected on disjoint fit nodes): **edge_lead2**

| path | text NLL | Δ vs diag | text acc | edge NLL | Δ vs diag | edge AUC | edge F1 |
|---|---|---|---|---|---|---|---|
| diagonal | 5.248 | +0.000±0.000 | 0.243 | 0.3847 | +0.000±0.000 | 0.790 | 0.123 |
| diagonal_random | 5.446 | +0.198±0.058 | 0.222 | 0.3914 | +0.007±0.006 | 0.777 | 0.157 |
| text_first | 5.108 | -0.140±0.034 | 0.244 | 0.3810 | -0.004±0.005 | 0.797 | 0.174 |
| edge_first | 5.155 | -0.093±0.036 | 0.241 | 0.3955 | +0.011±0.005 | 0.770 | 0.162 |
| text_lead2 | 5.240 | -0.008±0.045 | 0.242 | 0.3833 | -0.001±0.005 | 0.793 | 0.129 |
| text_lead1 | 5.284 | +0.036±0.037 | 0.245 | 0.3808 | -0.004±0.004 | 0.797 | 0.169 |
| edge_lead1 | 5.153 | -0.095±0.034 | 0.244 | 0.3888 | +0.004±0.004 | 0.783 | 0.151 |
| edge_lead2 | 4.982 | -0.266±0.034 | 0.244 | 0.3930 | +0.008±0.005 | 0.772 | 0.182 |
| joint_conf | 5.227 | -0.022±0.032 | 0.241 | 0.3920 | +0.007±0.005 | 0.775 | 0.190 |
| policy_grpo | 5.197 | -0.051±0.035 | 0.243 | 0.3829 | -0.002±0.005 | 0.792 | 0.173 |

policy text share per step: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

| steps (denoiser calls) | diag text NLL | diag edge NLL | diag AUC | conf text NLL | conf edge NLL | conf AUC |
|---|---|---|---|---|---|---|
| 1 | 4.747 | 0.3977 | 0.766 | 4.747 | 0.3977 | 0.766 |
| 2 | 4.928 | 0.3864 | 0.783 | 4.826 | 0.3978 | 0.765 |
| 4 | 5.117 | 0.3862 | 0.787 | 5.065 | 0.3937 | 0.771 |
| 8 | 5.248 | 0.3847 | 0.790 | 5.227 | 0.3920 | 0.775 |
| 16 | 5.310 | 0.3837 | 0.793 | 5.351 | 0.3919 | 0.775 |

### citeseer (homophily 0.74) — rho=0.5, N=8, mode=teacher, test cold nodes=79 x 3 resamplings
tuned fixed path (selected on disjoint fit nodes): **diagonal_random**

| path | text NLL | Δ vs diag | text acc | edge NLL | Δ vs diag | edge AUC | edge F1 |
|---|---|---|---|---|---|---|---|
| diagonal | 3.850 | +0.000±0.000 | 0.333 | 0.3743 | +0.000±0.000 | 0.815 | 0.136 |
| diagonal_random | 3.828 | -0.021±0.033 | 0.339 | 0.3718 | -0.003±0.006 | 0.818 | 0.155 |
| text_first | 4.003 | +0.153±0.026 | 0.318 | 0.3579 | -0.016±0.005 | 0.840 | 0.162 |
| edge_first | 4.020 | +0.170±0.026 | 0.316 | 0.3893 | +0.015±0.005 | 0.782 | 0.140 |
| text_lead2 | 3.961 | +0.111±0.026 | 0.318 | 0.3594 | -0.015±0.005 | 0.839 | 0.193 |
| text_lead1 | 3.836 | -0.013±0.021 | 0.336 | 0.3639 | -0.010±0.004 | 0.833 | 0.179 |
| edge_lead1 | 3.956 | +0.106±0.021 | 0.326 | 0.3823 | +0.008±0.003 | 0.796 | 0.151 |
| edge_lead2 | 4.146 | +0.296±0.028 | 0.302 | 0.3877 | +0.013±0.005 | 0.783 | 0.160 |
| joint_conf | 3.892 | +0.042±0.021 | 0.331 | 0.3864 | +0.012±0.005 | 0.786 | 0.141 |
| policy_grpo | 3.895 | +0.046±0.024 | 0.330 | 0.3578 | -0.017±0.005 | 0.838 | 0.213 |

policy text share per step: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

| steps (denoiser calls) | diag text NLL | diag edge NLL | diag AUC | conf text NLL | conf edge NLL | conf AUC |
|---|---|---|---|---|---|---|
| 1 | 4.747 | 0.3977 | 0.766 | 4.747 | 0.3977 | 0.766 |
| 2 | 4.289 | 0.3783 | 0.799 | 4.418 | 0.3975 | 0.766 |
| 4 | 4.016 | 0.3765 | 0.809 | 4.104 | 0.3906 | 0.778 |
| 8 | 3.850 | 0.3743 | 0.815 | 3.892 | 0.3864 | 0.786 |
| 16 | 3.765 | 0.3716 | 0.820 | 3.785 | 0.3836 | 0.792 |

### citeseer (homophily 0.74) — rho=1.0, N=8, mode=generate, test cold nodes=79 x 3 resamplings
tuned fixed path (selected on disjoint fit nodes): **edge_first**

| path | text NLL | Δ vs diag | text acc | edge NLL | Δ vs diag | edge AUC | edge F1 |
|---|---|---|---|---|---|---|---|
| diagonal | 7.851 | +0.000±0.000 | 0.040 | 0.5058 | +0.000±0.000 | 0.514 | 0.009 |
| diagonal_random | 8.687 | +0.836±0.060 | 0.032 | 0.5089 | +0.003±0.011 | 0.541 | 0.038 |
| text_first | 7.649 | -0.202±0.035 | 0.045 | 0.5556 | +0.050±0.015 | 0.530 | 0.004 |
| edge_first | 7.673 | -0.178±0.038 | 0.044 | 0.4893 | -0.017±0.006 | 0.527 | 0.004 |
| text_lead2 | 7.825 | -0.025±0.039 | 0.046 | 0.5483 | +0.042±0.015 | 0.498 | 0.004 |
| text_lead1 | 7.898 | +0.047±0.037 | 0.043 | 0.5208 | +0.015±0.009 | 0.519 | 0.017 |
| edge_lead1 | 7.857 | +0.006±0.036 | 0.039 | 0.4960 | -0.010±0.005 | 0.527 | 0.026 |
| edge_lead2 | 7.682 | -0.169±0.040 | 0.042 | 0.4925 | -0.013±0.006 | 0.531 | 0.021 |
| joint_conf | 7.837 | -0.014±0.042 | 0.039 | 0.4892 | -0.017±0.006 | 0.530 | 0.004 |

| steps (denoiser calls) | diag text NLL | diag edge NLL | diag AUC | conf text NLL | conf edge NLL | conf AUC |
|---|---|---|---|---|---|---|
| 1 | 7.128 | 0.4888 | 0.530 | 7.128 | 0.4888 | 0.530 |
| 2 | 7.430 | 0.4975 | 0.527 | 7.367 | 0.4888 | 0.530 |
| 4 | 7.629 | 0.4992 | 0.522 | 7.540 | 0.4890 | 0.530 |
| 8 | 7.851 | 0.5058 | 0.514 | 7.837 | 0.4892 | 0.530 |
| 16 | 8.320 | 0.5023 | 0.521 | 8.223 | 0.4915 | 0.528 |

### cora (homophily 0.81) — rho=0.5, N=8, mode=generate, test cold nodes=67 x 3 resamplings
tuned fixed path (selected on disjoint fit nodes): **edge_lead2**

| path | text NLL | Δ vs diag | text acc | edge NLL | Δ vs diag | edge AUC | edge F1 |
|---|---|---|---|---|---|---|---|
| diagonal | 4.817 | +0.000±0.000 | 0.327 | 0.4973 | +0.000±0.000 | 0.745 | 0.300 |
| diagonal_random | 5.124 | +0.307±0.085 | 0.298 | 0.5051 | +0.008±0.011 | 0.739 | 0.279 |
| text_first | 4.667 | -0.150±0.057 | 0.331 | 0.5176 | +0.020±0.012 | 0.738 | 0.272 |
| edge_first | 4.688 | -0.129±0.064 | 0.322 | 0.4951 | -0.002±0.009 | 0.741 | 0.284 |
| text_lead2 | 4.868 | +0.051±0.067 | 0.325 | 0.5194 | +0.022±0.012 | 0.727 | 0.299 |
| text_lead1 | 4.920 | +0.103±0.063 | 0.325 | 0.5004 | +0.003±0.008 | 0.744 | 0.308 |
| edge_lead1 | 4.697 | -0.120±0.065 | 0.326 | 0.4917 | -0.006±0.007 | 0.746 | 0.304 |
| edge_lead2 | 4.547 | -0.270±0.058 | 0.326 | 0.4938 | -0.004±0.008 | 0.740 | 0.291 |
| joint_conf | 4.779 | -0.038±0.060 | 0.321 | 0.4915 | -0.006±0.008 | 0.743 | 0.320 |
| policy_grpo | 4.794 | -0.023±0.047 | 0.326 | 0.4922 | -0.005±0.008 | 0.745 | 0.307 |

policy text share per step: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

| steps (denoiser calls) | diag text NLL | diag edge NLL | diag AUC | conf text NLL | conf edge NLL | conf AUC |
|---|---|---|---|---|---|---|
| 1 | 4.243 | 0.4966 | 0.736 | 4.243 | 0.4966 | 0.736 |
| 2 | 4.463 | 0.4957 | 0.740 | 4.347 | 0.4963 | 0.736 |
| 4 | 4.656 | 0.4954 | 0.746 | 4.582 | 0.4928 | 0.740 |
| 8 | 4.817 | 0.4973 | 0.745 | 4.779 | 0.4915 | 0.743 |
| 16 | 4.927 | 0.4969 | 0.747 | 4.964 | 0.4910 | 0.745 |

### cora (homophily 0.81) — rho=0.5, N=8, mode=teacher, test cold nodes=67 x 3 resamplings
tuned fixed path (selected on disjoint fit nodes): **text_lead1**

| path | text NLL | Δ vs diag | text acc | edge NLL | Δ vs diag | edge AUC | edge F1 |
|---|---|---|---|---|---|---|---|
| diagonal | 3.309 | +0.000±0.000 | 0.422 | 0.4882 | +0.000±0.000 | 0.761 | 0.310 |
| diagonal_random | 3.367 | +0.057±0.035 | 0.423 | 0.4800 | -0.008±0.011 | 0.767 | 0.296 |
| text_first | 3.480 | +0.170±0.036 | 0.406 | 0.4788 | -0.009±0.011 | 0.782 | 0.324 |
| edge_first | 3.523 | +0.213±0.034 | 0.398 | 0.4955 | +0.007±0.009 | 0.745 | 0.244 |
| text_lead2 | 3.406 | +0.097±0.035 | 0.408 | 0.4730 | -0.015±0.010 | 0.780 | 0.331 |
| text_lead1 | 3.316 | +0.007±0.029 | 0.421 | 0.4772 | -0.011±0.008 | 0.774 | 0.327 |
| edge_lead1 | 3.441 | +0.131±0.030 | 0.405 | 0.4948 | +0.007±0.008 | 0.749 | 0.290 |
| edge_lead2 | 3.623 | +0.313±0.033 | 0.393 | 0.4935 | +0.005±0.009 | 0.747 | 0.251 |
| joint_conf | 3.409 | +0.099±0.033 | 0.414 | 0.4953 | +0.007±0.008 | 0.745 | 0.268 |
| policy_grpo | 3.378 | +0.069±0.029 | 0.411 | 0.4746 | -0.014±0.011 | 0.782 | 0.331 |

policy text share per step: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

| steps (denoiser calls) | diag text NLL | diag edge NLL | diag AUC | conf text NLL | conf edge NLL | conf AUC |
|---|---|---|---|---|---|---|
| 1 | 4.243 | 0.4966 | 0.736 | 4.243 | 0.4966 | 0.736 |
| 2 | 3.804 | 0.4917 | 0.748 | 3.991 | 0.4983 | 0.735 |
| 4 | 3.497 | 0.4854 | 0.760 | 3.616 | 0.4905 | 0.746 |
| 8 | 3.309 | 0.4882 | 0.761 | 3.409 | 0.4953 | 0.745 |
| 16 | 3.237 | 0.4866 | 0.762 | 3.257 | 0.4931 | 0.751 |

### cora (homophily 0.81) — rho=1.0, N=8, mode=generate, test cold nodes=67 x 3 resamplings
tuned fixed path (selected on disjoint fit nodes): **edge_first**

| path | text NLL | Δ vs diag | text acc | edge NLL | Δ vs diag | edge AUC | edge F1 |
|---|---|---|---|---|---|---|---|
| diagonal | 8.321 | +0.000±0.000 | 0.056 | 0.6952 | +0.000±0.000 | 0.529 | 0.066 |
| diagonal_random | 9.586 | +1.266±0.129 | 0.049 | 0.6590 | -0.036±0.024 | 0.544 | 0.085 |
| text_first | 7.841 | -0.480±0.066 | 0.058 | 0.8850 | +0.190±0.027 | 0.509 | 0.023 |
| edge_first | 7.806 | -0.515±0.065 | 0.058 | 0.6345 | -0.061±0.018 | 0.540 | 0.053 |
| text_lead2 | 7.900 | -0.421±0.069 | 0.063 | 0.8329 | +0.138±0.025 | 0.507 | 0.038 |
| text_lead1 | 8.299 | -0.022±0.073 | 0.056 | 0.7987 | +0.104±0.020 | 0.491 | 0.055 |
| edge_lead1 | 8.172 | -0.149±0.053 | 0.054 | 0.6529 | -0.042±0.017 | 0.530 | 0.065 |
| edge_lead2 | 7.844 | -0.476±0.059 | 0.056 | 0.6447 | -0.050±0.017 | 0.539 | 0.058 |
| joint_conf | 8.275 | -0.046±0.062 | 0.055 | 0.6382 | -0.057±0.017 | 0.536 | 0.053 |

| steps (denoiser calls) | diag text NLL | diag edge NLL | diag AUC | conf text NLL | conf edge NLL | conf AUC |
|---|---|---|---|---|---|---|
| 1 | 6.840 | 0.6348 | 0.538 | 6.840 | 0.6348 | 0.538 |
| 2 | 7.339 | 0.6776 | 0.518 | 7.243 | 0.6347 | 0.538 |
| 4 | 7.851 | 0.7056 | 0.509 | 7.778 | 0.6356 | 0.537 |
| 8 | 8.321 | 0.6952 | 0.529 | 8.275 | 0.6382 | 0.536 |
| 16 | 8.931 | 0.6874 | 0.529 | 8.734 | 0.6353 | 0.534 |

### pubmed (homophily 0.80) — rho=0.5, N=8, mode=generate, test cold nodes=400 x 3 resamplings
tuned fixed path (selected on disjoint fit nodes): **edge_lead2**

| path | text NLL | Δ vs diag | text acc | edge NLL | Δ vs diag | edge AUC | edge F1 |
|---|---|---|---|---|---|---|---|
| diagonal | 3.451 | +0.000±0.000 | 0.503 | 0.3094 | +0.000±0.000 | 0.893 | 0.412 |
| diagonal_random | 3.806 | +0.355±0.036 | 0.467 | 0.3086 | -0.001±0.004 | 0.891 | 0.413 |
| text_first | 3.282 | -0.169±0.021 | 0.501 | 0.3041 | -0.005±0.003 | 0.894 | 0.432 |
| edge_first | 3.258 | -0.193±0.022 | 0.503 | 0.3045 | -0.005±0.003 | 0.894 | 0.435 |
| text_lead2 | 3.571 | +0.120±0.026 | 0.492 | 0.3026 | -0.007±0.003 | 0.896 | 0.427 |
| text_lead1 | 3.539 | +0.088±0.021 | 0.502 | 0.3021 | -0.007±0.002 | 0.896 | 0.438 |
| edge_lead1 | 3.305 | -0.146±0.019 | 0.503 | 0.3130 | +0.004±0.002 | 0.888 | 0.410 |
| edge_lead2 | 3.117 | -0.334±0.023 | 0.501 | 0.3131 | +0.004±0.002 | 0.886 | 0.418 |
| joint_conf | 3.387 | -0.064±0.018 | 0.504 | 0.3081 | -0.001±0.002 | 0.891 | 0.427 |
| policy_grpo | 3.403 | -0.048±0.019 | 0.503 | 0.3055 | -0.004±0.003 | 0.894 | 0.441 |

policy text share per step: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

| steps (denoiser calls) | diag text NLL | diag edge NLL | diag AUC | conf text NLL | conf edge NLL | conf AUC |
|---|---|---|---|---|---|---|
| 1 | 2.966 | 0.3074 | 0.890 | 2.966 | 0.3074 | 0.890 |
| 2 | 3.069 | 0.3034 | 0.896 | 2.968 | 0.3087 | 0.889 |
| 4 | 3.264 | 0.3051 | 0.896 | 3.162 | 0.3054 | 0.893 |
| 8 | 3.451 | 0.3094 | 0.893 | 3.387 | 0.3081 | 0.891 |
| 16 | 3.570 | 0.3080 | 0.895 | 3.563 | 0.3093 | 0.890 |
