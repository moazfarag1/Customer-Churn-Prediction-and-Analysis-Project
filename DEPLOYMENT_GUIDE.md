# 🚀 الدليل الشامل لنشر المشروع (DEPLOYMENT GUIDE)

هذا التقرير مخصص لتوضيح كيفية تجهيز مشروع **Customer Churn Prediction** والنشر النهائي على منصتي **GitHub** و **Streamlit Community Cloud**.

---

## 1. الملفات التي يجب رفعها على GitHub
بناءً على فحص المشروع بالكامل، يجب رفع الملفات التالية:
- **الملفات البرمجية والتطبيق:** `app/app.py`، `runtime_audit_utils.py`
- **النماذج (Models):** `models/final_model_pipeline.pkl` و `models/feature_schema.pkl` وجميع ملفات `.pkl` الأخرى. (الحجم الكلي صغير جداً حوالي 12MB كحد أقصى، ولا يتجاوز حد 100MB الخاص بـ GitHub، لذلك لا نحتاج لـ Git LFS).
- **البيانات (Data):** `data/raw/*.csv` و `data/cleaned/*.csv`. (لأن حجم البيانات لا يتجاوز 2 ميجابايت ولا تحتوي على بيانات حساسة PII، فمن الأفضل رفعها لجعل المشروع قابلاً لإعادة التشغيل بالكامل Reproducible).
- **الرسوم البيانية والصور:** فولدر `assets/plots/` بكل محتوياته، لأن التطبيق يستدعيها في لوحة التحكم (Dashboard).
- **دفاتر Jupyter:** فولدر `notebooks/` بكافة الدفاتر من 01 إلى 04.
- **التوثيقات:** فولدر `docs/`، `reports/`، وملفات `README.md`، `MODEL_CARD.md`، و `LICENSE`.
- **ملفات إعداد النشر:** `requirements.txt`، `packages.txt`، `runtime.txt`.

**السبب:** هذه الملفات تشكل بيئة العمل الكاملة. التطبيق يحتاج النماذج والصور ليعمل، ويحتاج ملفات الـ requirements لتثبيت المكتبات، بينما الدفاتر والتقارير توثق عمل الفريق الاحترافي.

---

## 2. الملفات التي لا يجب رفعها (تجاهلها)
- `.venv/`, `venv/`, `env/`: بيئة بايثون الوهمية (تثبت محلياً ولا تُنقل لأنها تختلف من نظام لآخر).
- `__pycache__/`, `*.pyc`: ملفات بايثون المؤقتة والمترجمة (تُولد تلقائياً).
- `.ipynb_checkpoints/`: ملفات الحفظ التلقائي لـ Jupyter Notebooks (ليس لها فائدة على السيرفر).
- `temp/`, `logs/`: ملفات السجلات والملفات المؤقتة.
- `.DS_Store`, `Thumbs.db`: ملفات النظام التشغيلي (خاصة بـ Mac و Windows ولا علاقة لها بالكود).
- `secrets.toml` أو أي API Keys: (إن وجدت) لعدم تعريض حساباتك للخطر.

**السبب:** هذه الملفات خاصة بالبيئة المحلية للمطور، رفعها يزيد مساحة المستودع بلا فائدة، وقد يسبب تعارضات، والأهم أنه قد يكشف معلومات حساسة (Credentials).

---

## 3. الهيكل النهائي للمشروع (Repository Structure)
الهيكل المثالي والاحترافي والموجود حالياً والذي ننصح بالالتزام به:
```text
project_root/
│
├── app/                  # يحتوي على تطبيق Streamlit (app.py)
├── assets/               # الصور والرسوم البيانية المستخدمة في التطبيق
├── data/                 # البيانات الخام والمعالجة والملخصات
├── docs/                 # توثيقات المشروع الإدارية والتنفيذية
├── models/               # نماذج الـ Machine Learning بصيغة (pkl) و (Schema)
├── notebooks/            # دفاتر Jupyter لمراحل التحليل والتطوير
├── reports/              # التقارير الفنية للنماذج المختلفة
├── .gitignore            # ملف تجاهل الملفات غير المطلوبة
├── DEPLOYMENT_GUIDE.md   # هذا الملف
├── LICENSE               # رخصة الاستخدام
├── README.md             # واجهة المشروع والتعريف به
├── requirements.txt      # مكتبات بايثون الأساسية (Pandas, Streamlit, sklearn...)
├── packages.txt          # مكتبات النظام لـ Streamlit Cloud (فارغ حالياً)
└── runtime.txt           # تحديد إصدار بايثون (python-3.11.8)
```

---

## 4. ملف `.gitignore` الاحترافي
تم إنشاء وتحديث ملف `.gitignore` كالتالي:
```text
# Environments
.env
.venv/
venv/
env/
ENV/

# Python caches and compiled files
__pycache__/
*.py[cod]
*$py.class
.pytest_cache/

# Jupyter Notebook
.ipynb_checkpoints/

# OS generated files
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db
desktop.ini

# IDEs
.idea/
.vscode/
*.swp
*.swo

# Logs and databases
*.log
*.sqlite
logs/

# Temporary and build files
temp/
tmp/
build/
dist/
*.egg-info/

# Secrets & Configs
secrets.toml
credentials/
```

---

## 5. دليل الرفع على GitHub (خطوة بخطوة)
1. **إنشاء Repository جديد:**
   - افتح [GitHub](https://github.com/) وسجل الدخول.
   - اضغط على زر **New** لإنشاء مستودع جديد.
   - اكتب اسم المشروع (مثلاً `Customer-Churn-Predictor`).
   - اجعله **Public** ولا تضف `README` أو `.gitignore` من المنصة (لأنها موجودة لديك).
2. **ربط الكود المحلي بـ GitHub:**
   - افتح موجه الأوامر (Terminal أو PowerShell) داخل مجلد المشروع.
   - نفذ الأوامر التالية بالترتيب:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Production ready customer churn app"
   git branch -M main
   git remote add origin https://github.com/<YOUR_USERNAME>/<YOUR_REPO_NAME>.git
   git push -u origin main
   ```
3. **كيفية التحديث مستقبلاً:**
   - إذا عدلت أي ملف مستقبلاً وتريد رفعه:
   ```bash
   git add .
   git commit -m "وصف التعديل هنا"
   git push
   ```

---

## 6. دليل النشر على Streamlit Community Cloud
1. **إنشاء الحساب والربط:**
   - اذهب إلى [share.streamlit.io](https://share.streamlit.io/).
   - سجل الدخول باستخدام حساب **GitHub** الخاص بك (سيتم الربط تلقائياً لتمكين المنصة من قراءة الكود).
2. **بدء النشر (New App):**
   - اضغط على زر **New app**.
3. **تحديد الإعدادات:**
   - **Repository:** ابحث واختر مستودع المشروع الخاص بك (مثال: `moazfarag1/Customer-Churn-Prediction-and-Analysis-Project`).
   - **Branch:** اختر `main`.
   - **Main file path:** اكتب المسار الدقيق لملف التطبيق: `app/app.py`.
4. **النشر والتثبيت:**
   - اضغط على **Deploy!**
   - ستقوم المنصة تلقائياً بقراءة `runtime.txt` لتثبيت بايثون، و `requirements.txt` لتثبيت المكتبات، وستبدأ تشغيل التطبيق.
5. **مراقبة السجلات (Logs):**
   - أثناء النشر، سترى شاشة زرقاء/سوداء في الزاوية السفلى (Manage app)، يمكنك من خلالها قراءة السجلات.
   - إذا حدث خطأ في مكتبة معينة، سيظهر هناك.

---

## 7. المشاكل المحتملة وحلولها
1. **مشكلة:** `ModuleNotFoundError: No module named 'xgboost'`
   - **الحل:** التأكد من وجود `xgboost` داخل `requirements.txt`. (تمت مراجعة الملف وهو موجود).
2. **مشكلة:** `FileNotFoundError` بسبب مسارات محلية (مثل `C:\Users\...`).
   - **الحل:** تم فحص الكود في `app/app.py` ووجدنا أن المبرمج استخدم `pathlib.Path(__file__).parent.parent`، وهذا هو الحل الاحترافي والديناميكي الذي يعمل على السيرفر بدون أخطاء.
3. **مشكلة:** `Critical Error: Model artifact not found`.
   - **الحل:** تأكد أنك لم تضع ملفات `.pkl` داخل مجلد تم تجاهله في `.gitignore`، تأكد أنها موجودة على GitHub داخل فولدر `models/`.

---

## 8. Checklist (قبل النشر)
- [x] الكود يعمل محلياً بكفاءة (`streamlit run app/app.py`).
- [x] المسارات داخل `app.py` ديناميكية (Relative Paths) ولا تعتمد على مسارات الكمبيوتر المحلي.
- [x] ملف `requirements.txt` يحتوي على `scikit-learn`, `pandas`, `streamlit`, `joblib`, إلخ.
- [x] ملفات الـ Models حجمها يسمح بالرفع العادي (أقل من 25 ميجا بايت).
- [x] ملف `.gitignore` يستبعد الملفات غير الضرورية والمؤقتة.

---

## 9. Checklist (بعد النشر)
- [ ] راجع الـ Logs وقت التشغيل للتأكد من عدم وجود أخطاء في الـ Dependencies.
- [ ] جرب تغيير إعدادات عميل افتراضي في التطبيق واضغط على (Predict).
- [ ] تأكد من ظهور الرسوم البيانية في تبويب (Dashboard) بشكل سليم.

---

## 10. التقييم النهائي: هل المشروع جاهز للنشر؟

✅ **التقييم: المشروع جاهز بنسبة 100% للنشر الفوري.**

- **من حيث الـ Paths:** الكود مكتوب بشكل ممتاز يعتمد على `pathlib`، وبالتالي لن يفشل في إيجاد النماذج أو الصور على سيرفرات لينكس.
- **من حيث الـ Dependencies:** جميع المكتبات المطلوبة (`streamlit`, `pandas`, `joblib`, `scikit-learn`, `numpy`, إلخ) موجودة بالفعل في `requirements.txt`.
- **من حيث الـ Artifacts:** تم توليد ملفات `runtime.txt` و `packages.txt` لتوفير أقصى درجات الاستقرار لمنصة Streamlit Cloud.
- **التعديلات التي تم إجراؤها:**
  - تمت إضافة `.gitignore` شامل واحترافي.
  - تم تعديل `README.md` ليشتمل على بنود Deployment و Screenshots و License لتظهر كواجهة احترافية.
  
المشروع بصورته الحالية يمثل **Best Practice** ويمكن رفعه فوراً بنجاح.
