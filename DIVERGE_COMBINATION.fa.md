# ترکیب TwinMask با DIVERGE: روش، آزمایش‌ها و نتایج

این گزارش فقط ترکیب‌های موجود در همین پروژه را توضیح می‌دهد. نتایج از فایل‌های ذخیره‌شده استخراج شده‌اند و آزمایش تازه‌ای اجرا نشده است. جدول‌های انتهای فایل میانگین و انحراف معیار اجراهای موجود را نشان می‌دهند.

## ایدهٔ ترکیب چیست؟

DIVERGE از چند نمایش متنی حاصل از initializationهای متفاوت LoRA استفاده می‌کند؛ برای هر نمایش یک GNN آموزش می‌دهد و پیش‌بینی‌ها را ترکیب می‌کند. TwinMask در این ترکیب منبع دیگری برای تولید نمایش‌های متنوع است: تغییر نحوهٔ خراب‌کردن و بازسازی متن و ساختار، نمایش متفاوتی از همان نودها می‌سازد.

ترکیب پیاده‌سازی‌شده در سطح نما و پیش‌بینی است. برای هر نما طبقه‌بند جداگانه آموزش داده می‌شود و سپس احتمال کلاس‌ها ترکیب می‌شود. این آزمایش‌ها آموزش مشترک end-to-end یک LLM و TwinMask نیستند. همچنین بخش اصلی این پل از diffusion سطح embedding در فاز صفر استفاده می‌کند؛ نباید آن را با ادغام مستقیم مدل توکنی DistilRoBERTa اشتباه گرفت.

## ۱. پل اولیه: MiniLM و diffusion در کنار DIVERGE

پیاده‌سازی: [diverge_bridge.py](scripts/diverge_bridge.py) و [tune_bridge.py](scripts/tune_bridge.py).

متن نودها ابتدا با MiniLM ثابت به بردار تبدیل می‌شود. چهار denoiser با تنظیمات joint، tied، text_only و edge_only روی این بردارها آموزش می‌بینند. همراه با بردار خام MiniLM، پنج نما به دست می‌آید. یک SAGE برای هر نما آموزش می‌بیند.

در سمت DIVERGE، پنج نمایش حاصل از LoRA با initializationهای PiSSA، Gaussian، EVA، LoftQ و Orthogonal و طبقه‌بندهای مربوط استفاده می‌شوند. سه مجموعه قابل مقایسه‌اند: پنج عضو DIVERGE، پنج عضو MiniLM و diffusion، و اجتماع ده عضو. ترکیب uniform میانگین احتمال‌هاست؛ ترکیب learned از WeightLearner استفاده می‌کند.

این بخش از تقسیم ۶۰/۲۰/۲۰ و seedهای ۴۲ تا ۴۵ استفاده می‌کند. نسخهٔ تیون‌شده، تنظیم diffusion، طبقه‌بندها و ensemble را با validation انتخاب می‌کند. خروجی اولیه و خروجی تیون‌شده دو آزمایش متفاوت‌اند و در جدول‌ها جدا هستند.

در گزارش موجود، ensemble نماهای MiniLM و diffusion تقریباً با DIVERGE هم‌سطح است: حدود ۸۸٫۶۵ در مقابل ۸۸٫۸۸ درصد روی Cora و ۷۸٫۲۵ در مقابل ۷۸٫۲۱ روی CiteSeer. ترکیب learned دو مجموعه حدود ۸۹٫۵۸ و ۷۹٫۰۸ درصد گزارش شده است. جدول‌های استخراج‌شدهٔ زیر جزئیات هر نوع ensemble را جدا نشان می‌دهند. افزایش میانگین به‌تنهایی اثبات برتری آماری نیست.

در این مسیر، استخراج نمایش MiniLM و diffusion برچسب کلاس نمی‌بیند، اما طبقه‌بند SAGE و WeightLearner از برچسب استفاده می‌کنند. در مقابل، نمایش‌های LoRA متعلق به DIVERGE خودشان نیز با نظارت برچسب آموزش دیده‌اند.

## ۲. ترکیب با افزایش دادهٔ نیمه‌نظارتی DIVERGE

پیاده‌سازی: [semi_aug_bridge.py](scripts/semi_aug_bridge.py).

در این مسیر ورودی diffusion، embedding حاصل از LoRA است. پنج نمای DIVERGE حفظ می‌شوند. روی نمای PiSSA چهار نسخهٔ joint، tied، text_only و edge_only و روی هر یک از چهار initialization دیگر یک joint آموزش داده می‌شود؛ در مجموع هشت نمای diffusion و سیزده عضو برای ترکیب وجود دارد.

برای هر عضو SAGE جداگانه آموزش داده می‌شود. آزمایش میانگین احتمال پنج عضو DIVERGE، هشت عضو diffusion و اجتماع سیزده عضو را مقایسه می‌کند. نسخهٔ screened نیز در فایل‌های نتیجه موجود است. انتخاب‌های روش باید با validation انجام شوند.

بازوهای آزمایش شامل دادهٔ اصلی و نسخه‌های افزوده هستند؛ نام واقعی هر بازو و تعداد اجراهای موجود در جدول‌های زیر ثبت شده است. در بازوی افزوده، نودهای مصنوعی با یال‌های انتخاب‌شده به گراف افزوده می‌شوند و می‌توانند به‌عنوان نمونه‌های آموزشی با برچسب شبه‌ای وارد شوند. این پروتکل public low-label است و با تقسیم ۶۰/۲۰/۲۰ بخش قبلی مقایسهٔ مستقیم ندارد.

اینجا فقط تابع هزینهٔ diffusion بدون برچسب است؛ کل خط لوله بدون برچسب نیست، چون embedding ورودی از LoRA آموزش‌دیده با برچسب آمده است. این تفاوت برای توصیف علمی روش ضروری است.

## ۳. ترکیب با روش‌های مختلف افزایش داده

پیاده‌سازی: [semi_methods_bridge.py](scripts/semi_methods_bridge.py).

همان پنج عضو DIVERGE و هشت عضو diffusion برای نه روش ref_plain، ref_selected، com_vote، own_attn، own_viewdis، lit_conf، diva2_ens، diva_mi و diva_full مقایسه شده‌اند. سازندهٔ هر روش از بستهٔ خارجی `tags_data_augmentation.semi_study.diva_ideas` وارد می‌شود. تعریف فیلتر، وزن، soft label و تغییر گراف در آن مخزن قرار دارد؛ این فایل فقط اجرای روش و اضافه‌کردن نماهای diffusion را انجام می‌دهد.

بنابراین این جدول‌ها پاسخ می‌دهند که «اضافه‌کردن diffusion به هر روش چه تغییری می‌دهد»، نه اینکه diffusion به‌تنهایی مسئول تمام عملکرد روش است. مقایسهٔ مناسب برای هر ردیف، ستون ترکیبی با ستون DIVERGE همان ردیف و همان seedهاست.

طبق گزارش پروژه، بهبود میانگین در نه روش حدود ۰٫۲۷ واحد درصد برای Cora و ۰٫۲۵ برای CiteSeer است. بعضی روش‌ها افت دارند و بعضی بهبود؛ افزایش‌ها عموماً کوچک‌اند. برای ادعای برتری هر روش، آزمون جفت‌شده و تکرار بیشتر لازم است. بهترین میانگین ترکیبی Cora در گزارش مربوط به own_attn و حدود ۸۵٫۸۵ درصد است.

## ۴. ترکیب با انتخاب فعال CF-DIVE

[diverge_views.py](cf_dive/diverge_views.py) پنج نمای LoRA و هشت نمای diffusion را روی گراف افزودهٔ ref_selected آماده می‌کند. [combined.py](cf_dive/combined.py) این نماها را برای انتخاب فعال مبتنی بر حساسیت ساختاری استفاده می‌کند. در این مسیر نودهای قابل انتخاب از بخش مجزایی انتخاب می‌شوند تا با مجموعهٔ آموزش عمومی LoRA هم‌پوشانی نداشته باشند.

کد این ترکیب وجود دارد، اما در خروجی‌های بررسی‌شده گزارش نهایی مستقل برای آن پیدا نشد. نتایج نسخهٔ Efficient دو عضوی CF-DIVE را نمی‌توان به این ترکیب سیزده‌عضوی نسبت داد. صرف وجود کد یا cache نیز اجرای موفق آزمایش کامل را اثبات نمی‌کند.

## نتیجه‌ای که داده‌های فعلی پشتیبانی می‌کنند

diffusion می‌تواند نماهای مکمل برای ensemble DIVERGE تولید کند. در پل MiniLM، این نماها با هزینهٔ نمایش متنی کوچک‌تر به عملکرد نزدیک DIVERGE می‌رسند و اجتماع دو مجموعه میانگین بهتری دارد. در مسیر نیمه‌نظارتی مبتنی بر LoRA، سود افزودن diffusion کوچک‌تر و وابسته به روش است. داده‌های فعلی برای ادعای برتری عمومی و قطعی ترکیب کافی نیستند.

هزینهٔ اجرا نیز باید همراه دقت بررسی شود: پنج یا هشت نمای جدید به معنی آموزش denoiser و طبقه‌بندهای بیشتر است. این گزارش هزینهٔ تازه‌ای اندازه‌گیری نکرده است. وابستگی به فایل‌ها و embeddingهای مخزن خارجی DIVERGE نیز برای بازتولید لازم است؛ فایل‌های اجرایی برخی مسیرهای پیش‌فرض محلی دارند که ممکن است نیاز به تنظیم داشته باشند.

## نتایج استخراج‌شده از JSON

اعداد accuracy برحسب درصد هستند. ± انحراف معیار روی اجراهای ذخیره‌شده است و بازهٔ اطمینان نیست. ستون افزایش ترکیب نسبت به DIVERGE برحسب واحد درصد است. جدول‌های اولیه، تیون‌شده و نیمه‌نظارتی را باید جدا تفسیر کرد.

### results/diverge_bridge/citeseer.json

Source: [citeseer.json](results/diverge_bridge/citeseer.json)

| Method / arm | Metric | Runs | Accuracy (%) |
|---|---|---:|---:|
| all | ENS_diverge5 (5 LoRA inits) | 4 | 78.21 +/- 1.64 |
| all | ENS_minilm+4 diffusion views | 4 | 77.31 +/- 1.86 |
| all | ENS_minilm raw + 4 diffusion views | 4 | 77.43 +/- 1.89 |
| all | ENS_gemma_pissa+4 diffusion views | 4 | 77.63 +/- 1.78 |
| all | ENS_gemma_pissa raw + 4 diffusion views | 4 | 77.59 +/- 1.95 |
| all | ENS_diverge5 + gemma diffusion views | 4 | 78.06 +/- 2.02 |

### results/diverge_bridge/cora.json

Source: [cora.json](results/diverge_bridge/cora.json)

| Method / arm | Metric | Runs | Accuracy (%) |
|---|---|---:|---:|
| all | ENS_diverge5 (5 LoRA inits) | 4 | 88.88 +/- 1.40 |
| all | ENS_minilm+4 diffusion views | 4 | 88.33 +/- 0.30 |
| all | ENS_minilm raw + 4 diffusion views | 4 | 88.65 +/- 0.62 |
| all | ENS_gemma_pissa+4 diffusion views | 4 | 88.33 +/- 1.38 |
| all | ENS_gemma_pissa raw + 4 diffusion views | 4 | 88.51 +/- 1.17 |
| all | ENS_diverge5 + gemma diffusion views | 4 | 88.47 +/- 1.21 |

### results/diverge_bridge_tuned/citeseer.json

Source: [citeseer.json](results/diverge_bridge_tuned/citeseer.json)

| Method / arm | Metric | Runs | Accuracy (%) |
|---|---|---:|---:|
| all | ENS uniform  DIVERGE-5 | 4 | 78.21 +/- 1.64 |
| all | ENS learned  DIVERGE-5 | 4 | 78.10 +/- 2.03 |
| all | ENS uniform  MiniLM+4 diffusion (tuned) | 4 | 78.25 +/- 1.17 |
| all | ENS learned  MiniLM+4 diffusion (tuned) | 4 | 78.06 +/- 1.20 |
| all | ENS learned  DIVERGE-5 + our 5 views | 4 | 79.08 +/- 1.68 |

### results/diverge_bridge_tuned/cora.json

Source: [cora.json](results/diverge_bridge_tuned/cora.json)

| Method / arm | Metric | Runs | Accuracy (%) |
|---|---|---:|---:|
| all | ENS uniform  DIVERGE-5 | 4 | 88.88 +/- 1.40 |
| all | ENS learned  DIVERGE-5 | 4 | 88.70 +/- 1.29 |
| all | ENS uniform  MiniLM+4 diffusion (tuned) | 4 | 88.65 +/- 0.71 |
| all | ENS learned  MiniLM+4 diffusion (tuned) | 4 | 88.47 +/- 0.66 |
| all | ENS learned  DIVERGE-5 + our 5 views | 4 | 89.58 +/- 0.55 |

### results/semi_aug_bridge/citeseer.json

Source: [citeseer.json](results/semi_aug_bridge/citeseer.json)

| Method / arm | Metric | Runs | Accuracy (%) |
|---|---|---:|---:|
| base | ENS DIVERGE-1 (5 inits) | 3 | 71.60 +/- 0.68 |
| base | ENS diffusion (8 views) | 3 | 71.55 +/- 0.28 |
| base | ENS DIVERGE-1 + diffusion | 3 | 71.93 +/- 0.19 |
| base | ENS screened all views | 3 | 71.65 +/- 0.81 |
| augFull | ENS DIVERGE-1 (5 inits) | 3 | 73.92 +/- 0.10 |
| augFull | ENS diffusion (8 views) | 3 | 73.68 +/- 0.02 |
| augFull | ENS DIVERGE-1 + diffusion | 3 | 74.08 +/- 0.14 |
| augFull | ENS screened all views | 3 | 73.50 +/- 0.19 |

### results/semi_aug_bridge/cora.json

Source: [cora.json](results/semi_aug_bridge/cora.json)

| Method / arm | Metric | Runs | Accuracy (%) |
|---|---|---:|---:|
| base | ENS DIVERGE-1 (5 inits) | 3 | 84.48 +/- 0.51 |
| base | ENS diffusion (8 views) | 3 | 82.79 +/- 0.32 |
| base | ENS DIVERGE-1 + diffusion | 3 | 83.67 +/- 0.19 |
| base | ENS screened all views | 3 | 84.07 +/- 0.29 |
| augFull | ENS DIVERGE-1 (5 inits) | 3 | 84.75 +/- 0.44 |
| augFull | ENS diffusion (8 views) | 3 | 85.09 +/- 0.29 |
| augFull | ENS DIVERGE-1 + diffusion | 3 | 85.49 +/- 0.34 |
| augFull | ENS screened all views | 3 | 83.67 +/- 1.01 |

### results/semi_aug_bridge/cora_naive_knn.json

Source: [cora_naive_knn.json](results/semi_aug_bridge/cora_naive_knn.json)

| Method / arm | Metric | Runs | Accuracy (%) |
|---|---|---:|---:|
| base | ENS DIVERGE-1 (5 inits) | 3 | 83.95 +/- 0.28 |
| base | ENS diffusion (8 views) | 3 | 82.54 +/- 0.49 |
| base | ENS DIVERGE-1 + diffusion | 3 | 83.51 +/- 0.36 |
| base | ENS screened all views | 3 | 83.28 +/- 0.30 |
| augFull | ENS DIVERGE-1 (5 inits) | 3 | 73.16 +/- 1.55 |
| augFull | ENS diffusion (8 views) | 3 | 77.80 +/- 0.20 |
| augFull | ENS DIVERGE-1 + diffusion | 3 | 77.80 +/- 0.83 |
| augFull | ENS screened all views | 3 | 74.97 +/- 1.34 |

### results/semi_aug_bridge/methods_citeseer.json

Source: [methods_citeseer.json](results/semi_aug_bridge/methods_citeseer.json)

| Method / arm | Metric | Runs | Accuracy (%) |
|---|---|---:|---:|
| ref_plain | ENS DIVERGE-1 | 3 | 73.92 +/- 0.26 |
| ref_plain | ENS diffusion | 3 | 74.05 +/- 0.15 |
| ref_plain | ENS both | 3 | 74.19 +/- 0.24 |
| ref_plain | Combined minus DIVERGE (percentage points) | 3 | +0.27 |
| ref_selected | ENS DIVERGE-1 | 3 | 74.32 +/- 0.49 |
| ref_selected | ENS diffusion | 3 | 73.75 +/- 0.33 |
| ref_selected | ENS both | 3 | 73.94 +/- 0.16 |
| ref_selected | Combined minus DIVERGE (percentage points) | 3 | -0.38 |
| com_vote | ENS DIVERGE-1 | 3 | 73.69 +/- 0.46 |
| com_vote | ENS diffusion | 3 | 73.88 +/- 0.27 |
| com_vote | ENS both | 3 | 74.31 +/- 0.33 |
| com_vote | Combined minus DIVERGE (percentage points) | 3 | +0.61 |
| own_attn | ENS DIVERGE-1 | 3 | 74.08 +/- 0.22 |
| own_attn | ENS diffusion | 3 | 74.28 +/- 0.35 |
| own_attn | ENS both | 3 | 74.25 +/- 0.29 |
| own_attn | Combined minus DIVERGE (percentage points) | 3 | +0.17 |
| own_viewdis | ENS DIVERGE-1 | 3 | 74.06 +/- 0.08 |
| own_viewdis | ENS diffusion | 3 | 73.86 +/- 0.51 |
| own_viewdis | ENS both | 3 | 74.07 +/- 0.54 |
| own_viewdis | Combined minus DIVERGE (percentage points) | 3 | +0.01 |
| lit_conf | ENS DIVERGE-1 | 3 | 74.18 +/- 0.10 |
| lit_conf | ENS diffusion | 3 | 73.89 +/- 0.36 |
| lit_conf | ENS both | 3 | 74.19 +/- 0.18 |
| lit_conf | Combined minus DIVERGE (percentage points) | 3 | +0.01 |
| diva2_ens | ENS DIVERGE-1 | 3 | 74.37 +/- 0.07 |
| diva2_ens | ENS diffusion | 3 | 74.20 +/- 0.25 |
| diva2_ens | ENS both | 3 | 74.24 +/- 0.25 |
| diva2_ens | Combined minus DIVERGE (percentage points) | 3 | -0.13 |
| diva_mi | ENS DIVERGE-1 | 3 | 73.43 +/- 0.26 |
| diva_mi | ENS diffusion | 3 | 74.02 +/- 0.19 |
| diva_mi | ENS both | 3 | 74.21 +/- 0.13 |
| diva_mi | Combined minus DIVERGE (percentage points) | 3 | +0.78 |
| diva_full | ENS DIVERGE-1 | 3 | 72.80 +/- 0.17 |
| diva_full | ENS diffusion | 3 | 73.69 +/- 0.38 |
| diva_full | ENS both | 3 | 73.73 +/- 0.21 |
| diva_full | Combined minus DIVERGE (percentage points) | 3 | +0.94 |

### results/semi_aug_bridge/methods_cora.json

Source: [methods_cora.json](results/semi_aug_bridge/methods_cora.json)

| Method / arm | Metric | Runs | Accuracy (%) |
|---|---|---:|---:|
| ref_plain | ENS DIVERGE-1 | 3 | 85.15 +/- 0.65 |
| ref_plain | ENS diffusion | 3 | 84.77 +/- 0.58 |
| ref_plain | ENS both | 3 | 85.15 +/- 0.55 |
| ref_plain | Combined minus DIVERGE (percentage points) | 3 | +0.00 |
| ref_selected | ENS DIVERGE-1 | 3 | 85.15 +/- 0.28 |
| ref_selected | ENS diffusion | 3 | 84.93 +/- 0.49 |
| ref_selected | ENS both | 3 | 85.19 +/- 0.31 |
| ref_selected | Combined minus DIVERGE (percentage points) | 3 | +0.03 |
| com_vote | ENS DIVERGE-1 | 3 | 85.09 +/- 0.05 |
| com_vote | ENS diffusion | 3 | 85.06 +/- 0.36 |
| com_vote | ENS both | 3 | 85.43 +/- 0.13 |
| com_vote | Combined minus DIVERGE (percentage points) | 3 | +0.34 |
| own_attn | ENS DIVERGE-1 | 3 | 85.38 +/- 0.39 |
| own_attn | ENS diffusion | 3 | 85.12 +/- 0.22 |
| own_attn | ENS both | 3 | 85.85 +/- 0.43 |
| own_attn | Combined minus DIVERGE (percentage points) | 3 | +0.47 |
| own_viewdis | ENS DIVERGE-1 | 3 | 85.49 +/- 0.77 |
| own_viewdis | ENS diffusion | 3 | 85.06 +/- 0.12 |
| own_viewdis | ENS both | 3 | 85.54 +/- 0.64 |
| own_viewdis | Combined minus DIVERGE (percentage points) | 3 | +0.05 |
| lit_conf | ENS DIVERGE-1 | 3 | 85.43 +/- 0.51 |
| lit_conf | ENS diffusion | 3 | 85.17 +/- 0.54 |
| lit_conf | ENS both | 3 | 85.41 +/- 0.54 |
| lit_conf | Combined minus DIVERGE (percentage points) | 3 | -0.02 |
| diva2_ens | ENS DIVERGE-1 | 3 | 84.95 +/- 0.58 |
| diva2_ens | ENS diffusion | 3 | 84.86 +/- 0.14 |
| diva2_ens | ENS both | 3 | 85.17 +/- 0.36 |
| diva2_ens | Combined minus DIVERGE (percentage points) | 3 | +0.23 |
| diva_mi | ENS DIVERGE-1 | 3 | 83.82 +/- 0.38 |
| diva_mi | ENS diffusion | 3 | 84.11 +/- 0.16 |
| diva_mi | ENS both | 3 | 84.53 +/- 0.27 |
| diva_mi | Combined minus DIVERGE (percentage points) | 3 | +0.71 |
| diva_full | ENS DIVERGE-1 | 3 | 84.53 +/- 0.25 |
| diva_full | ENS diffusion | 3 | 84.78 +/- 0.61 |
| diva_full | ENS both | 3 | 85.14 +/- 0.55 |
| diva_full | Combined minus DIVERGE (percentage points) | 3 | +0.61 |
