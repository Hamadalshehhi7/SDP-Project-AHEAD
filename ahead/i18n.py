"""Patient-facing Arabic labels; clinical tools and generated reports remain English."""

import streamlit as st

AR = {
    "Overview": "الرئيسية", "Screenings": "الفحوصات", "Data & Analytics": "البيانات والتحليلات",
    "Clinical Dashboard": "لوحة الطبيب", "AI Assistant": "المساعد الذكي", "Settings": "الإعدادات",
    "Log Out": "تسجيل الخروج", "Open AHEAD Assistant": "افتح مساعد AHEAD",
    "Choose a screening": "اختر فحصاً", "Select a condition to begin a guided early-awareness assessment.": "اختر حالة صحية لبدء التقييم.",
    "Begin screening": "ابدأ الفحص", "Start screening": "ابدأ الفحص", "← Back to all screenings": "← جميع الفحوصات",
    "Prediction model": "نموذج التنبؤ", "Generate Screening Result": "اعرض نتيجة الفحص",
    "Screening Result": "نتيجة الفحص", "Recommended Next Steps": "الخطوات التالية",
    "Factors Identified From Your Inputs": "عوامل من بياناتك", "AHEAD Insight": "توضيح AHEAD",
    "Generate AHEAD Insight": "أنشئ التوضيح", "Elevated Risk Pattern": "مؤشرات تستدعي المراجعة",
    "Lower Predicted Risk": "مؤشرات أقل وفق النموذج", "Model screening score": "درجة النموذج",
    "Diabetes Screening": "فحص السكري", "Cardiovascular Screening": "فحص القلب والأوعية الدموية",
    "Kidney Health Screening": "فحص صحة الكلى", "Diabetes Risk Assessment": "تقييم خطر السكري",
    "Cardiovascular Risk Assessment": "تقييم خطر القلب والأوعية الدموية",
    "Kidney Health Assessment": "تقييم صحة الكلى", "Lifestyle & Patient Profile": "نمط الحياة وبيانات المريض",
    "Medical History": "التاريخ المرضي", "Clinical Measurements": "القياسات الطبية",
    "General Health": "الصحة العامة", "Family & Medical History": "تاريخ العائلة والمرض",
    "Yes": "نعم", "No": "لا", "Male": "ذكر", "Female": "أنثى",
    "Gender": "الجنس", "Age": "العمر", "Smoking History": "تاريخ التدخين",
    "BMI": "مؤشر كتلة الجسم", "Hypertension": "ارتفاع ضغط الدم",
    "Heart Disease": "أمراض القلب", "HbA1c Level": "السكر التراكمي",
    "Blood Glucose Level": "سكر الدم", "Sex": "الجنس", "Age Category": "الفئة العمرية",
    "Physical Activities": "النشاط البدني", "Sleep Hours": "ساعات النوم",
    "Smoker Status": "حالة التدخين", "Alcohol Drinkers": "استهلاك الكحول",
    "General Health": "الصحة العامة", "Physical Health Days": "أيام الصحة الجسدية",
    "Mental Health Days": "أيام الصحة النفسية", "Had Stroke": "جلطة سابقة",
    "Had Asthma": "الربو", "Had Kidney Disease": "مرض الكلى",
    "Had Arthritis": "التهاب المفاصل", "Had Diabetes": "السكري",
    "Systolic BP": "الضغط الانقباضي", "Diastolic BP": "الضغط الانبساطي",
    "Fasting Blood Sugar": "سكر الصيام", "Serum Creatinine": "كرياتينين الدم",
    "Protein in Urine": "بروتين البول", "Estimated GFR": "معدل ترشيح الكلى",
    "Model Interpretation": "شرح النموذج", "Your result: model input sensitivity": "تأثر نتيجة النموذج بالمدخلات",
    "View entered information": "عرض البيانات المدخلة", "Download screening PDF": "تحميل التقرير PDF",
    "Find a nearby clinic (map search)": "البحث عن عيادة قريبة على الخريطة",
}

AR_STEPS = {
    "diabetes": {
        1: [("راجع الطبيب", "ناقش النتيجة والأعراض والفحوصات السابقة مع طبيب الرعاية الأولية."),
            ("اسأل عن فحوصات التأكيد", "اسأل إن كانت هناك حاجة لإعادة فحص السكر التراكمي أو سكر الصيام."),
            ("راجع عوامل القلب", "اسأل عن قياس ضغط الدم والدهون وتقييم صحة القلب."),
            ("ضع خطة واقعية", "ناقش التغذية والنشاط والوزن مع المختص دون اتباع تغييرات شديدة.")],
        0: [("تابع الفحوصات المعتادة", "النتيجة الأقل لا تستبعد السكري. اتبع جدول الفحوصات الذي يوصي به طبيبك."),
            ("حافظ على عادات صحية", "واصل النشاط المنتظم والغذاء المتوازن والنوم الكافي.")],
    },
    "heart": {
        1: [("راجع صحة القلب", "حدد موعداً مع طبيب وناقش الحاجة إلى تقييم متخصص."),
            ("افحص عوامل الخطر", "ناقش ضغط الدم والدهون والسكر مع الطبيب."),
            ("اطلب دعماً للإقلاع عن التدخين", "إذا كنت تدخن، اسأل المختص عن المساعدة المناسبة."),
            ("ابدأ النشاط بتدرج", "ناقش النشاط المناسب لحالتك، وتوقف إذا ظهرت أعراض غير معتادة.")],
        0: [("واصل متابعة القلب", "النتيجة الأقل لا تستبعد المشكلات الحالية أو المستقبلية. تابع فحوصاتك المعتادة."),
            ("اهتم بالعوامل القابلة للتغيير", "تجنب التدخين واهتم بالنشاط والنوم والطعام المتوازن.")],
    },
    "kidney": {
        1: [("راجع صحة الكلى", "احجز موعداً لمراجعة النتائج والفحوصات السابقة مع الطبيب."),
            ("اسأل عن فحوصات الكلى", "ناقش الكرياتينين ومعدل الترشيح وفحص الألبومين في البول."),
            ("راجع الضغط والسكر", "اسأل إن كنت تحتاج متابعة إضافية لضغط الدم أو السكر."),
            ("راجع الأدوية", "أخبر الطبيب بالأدوية والمكملات، ولا توقف علاجاً موصوفاً بنفسك.")],
        0: [("واصل المتابعة المعتادة", "اسأل طبيبك عن مواعيد فحوصات الكلى إذا كانت لديك عوامل خطر."),
            ("احمِ صحة الكلى", "تابع الضغط والسكر وناقش استخدام المسكنات بانتظام مع المختص.")],
    },
}


def arabic():
    return st.session_state.get("pref_language") == "العربية"


def tr(value):
    return AR.get(str(value), str(value)) if arabic() else str(value)
