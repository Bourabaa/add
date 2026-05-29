from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_PATH = ROOT_DIR / "data" / "raw" / "organisations.csv"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "data" / "processed" / "organisations_enriched.csv"
OUTPUT_COLUMNS = (
    "organization",
    "organization_slug",
    "full_name",
    "aliases",
    "category",
    "domains",
    "keywords_fr",
    "keywords_en",
    "keywords_ar",
    "source_description",
    "profile_text",
    "package_count",
)


CATEGORY_PROFILES: dict[str, dict[str, list[str] | str]] = {
    "digital_transformation": {
        "domains": ["transformation digitale", "administration numerique", "innovation", "services numeriques"],
        "keywords_fr": ["digital", "numerique", "startup", "innovation", "e-gov", "transformation digitale", "services en ligne"],
        "keywords_en": ["digital", "digital government", "startup", "innovation", "e-government", "online services"],
        "keywords_ar": ["التحول الرقمي", "الادارة الرقمية", "الابتكار", "الخدمات الرقمية", "الحكومة الالكترونية", "الشركات الناشئة"],
    },
    "water_resources": {
        "domains": ["eau", "hydrologie", "bassin hydraulique", "ressources hydriques"],
        "keywords_fr": ["eau", "barrage", "pluie", "hydrologie", "nappe phreatique", "ressources hydriques"],
        "keywords_en": ["water", "hydrology", "rainfall", "dams", "groundwater", "water resources"],
        "keywords_ar": ["المياه", "الهيدرولوجيا", "السدود", "التساقطات", "الموارد المائية", "الفرشة المائية"],
    },
    "urban_planning": {
        "domains": ["urbanisme", "amenagement", "habitat", "planification urbaine"],
        "keywords_fr": ["urbanisme", "amenagement", "plan d amenagement", "habitat", "construction", "territoire"],
        "keywords_en": ["urban planning", "land use", "housing", "zoning", "territorial planning", "construction"],
        "keywords_ar": ["التعمير", "تهيئة المجال", "السكن", "التخطيط الحضري", "البناء", "التراب"],
    },
    "capital_markets": {
        "domains": ["finance", "marche des capitaux", "bourse", "titres financiers"],
        "keywords_fr": ["bourse", "marche des capitaux", "actions", "obligations", "finance", "investissement"],
        "keywords_en": ["capital markets", "stock exchange", "shares", "bonds", "finance", "investment"],
        "keywords_ar": ["سوق الرساميل", "البورصة", "الاسهم", "السندات", "التمويل", "الاستثمار"],
    },
    "telecom": {
        "domains": ["telecommunications", "internet", "reseaux", "frequences"],
        "keywords_fr": ["telecom", "internet", "mobile", "reseau", "frequences", "couverture"],
        "keywords_en": ["telecom", "internet", "mobile", "network", "spectrum", "coverage"],
        "keywords_ar": ["الاتصالات", "الانترنت", "الهاتف المحمول", "الشبكات", "الترددات", "التغطية"],
    },
    "archives_heritage": {
        "domains": ["archives", "documents historiques", "patrimoine documentaire"],
        "keywords_fr": ["archives", "documents", "memoire", "histoire", "patrimoine documentaire"],
        "keywords_en": ["archives", "historical records", "documents", "memory", "documentary heritage"],
        "keywords_ar": ["الارشيف", "الوثائق", "الذاكرة", "التاريخ", "التراث الوثائقي"],
    },
    "insurance_social_protection": {
        "domains": ["assurance", "prevoyance sociale", "protection sociale", "retraite"],
        "keywords_fr": ["assurance", "prevoyance sociale", "couverture", "retraite", "protection sociale"],
        "keywords_en": ["insurance", "social protection", "coverage", "pension", "social security"],
        "keywords_ar": ["التامين", "الحماية الاجتماعية", "التغطية", "التقاعد", "الضمان الاجتماعي"],
    },
    "central_bank": {
        "domains": ["banque centrale", "monnaie", "politique monetaire", "taux de change"],
        "keywords_fr": ["banque centrale", "monnaie", "inflation", "change", "credit", "taux directeur"],
        "keywords_en": ["central bank", "currency", "inflation", "exchange rate", "credit", "monetary policy"],
        "keywords_ar": ["البنك المركزي", "العملة", "التضخم", "سعر الصرف", "الائتمان", "السياسة النقدية"],
    },
    "retirement_pension": {
        "domains": ["retraite", "pension", "allocation", "cotisation"],
        "keywords_fr": ["retraite", "pension", "cotisation", "allocations", "regime de retraite"],
        "keywords_en": ["retirement", "pension", "contributions", "benefits", "pension scheme"],
        "keywords_ar": ["التقاعد", "المعاش", "المساهمات", "التعويضات", "نظام التقاعد"],
    },
    "investment_regional": {
        "domains": ["investissement", "entreprise", "territoire", "developpement regional"],
        "keywords_fr": ["investissement", "entreprise", "incitation", "projet", "territoire", "business regional"],
        "keywords_en": ["investment", "business", "incentives", "project", "regional development", "enterprise"],
        "keywords_ar": ["الاستثمار", "المقاولة", "التحفيز", "المشاريع", "التنمية الجهوية", "الاعمال"],
    },
    "commerce_industry_regional": {
        "domains": ["commerce", "industrie", "services", "entreprises regionales"],
        "keywords_fr": ["commerce", "industrie", "services", "entreprises", "activite economique"],
        "keywords_en": ["commerce", "industry", "services", "businesses", "economic activity"],
        "keywords_ar": ["التجارة", "الصناعة", "الخدمات", "المقاولات", "النشاط الاقتصادي"],
    },
    "social_protection": {
        "domains": ["protection sociale", "assurance maladie", "prestations sociales"],
        "keywords_fr": ["couverture sociale", "assurance maladie", "prestations", "beneficiaires", "securite sociale"],
        "keywords_en": ["social protection", "health insurance", "benefits", "beneficiaries", "social security"],
        "keywords_ar": ["الحماية الاجتماعية", "التامين الصحي", "التعويضات", "المستفيدون", "الضمان الاجتماعي"],
    },
    "local_government": {
        "domains": ["collectivites territoriales", "commune", "services locaux", "gouvernance locale"],
        "keywords_fr": ["commune", "collectivite", "budget local", "services communaux", "territoire"],
        "keywords_en": ["municipality", "local government", "local budget", "municipal services", "territory"],
        "keywords_ar": ["الجماعات الترابية", "البلدية", "الميزانية المحلية", "الخدمات الجماعية", "التراب"],
    },
    "agriculture": {
        "domains": ["agriculture", "cultures", "irrigation agricole", "developpement rural"],
        "keywords_fr": ["agriculture", "culture", "irrigation", "production agricole", "rural", "eaux et forets"],
        "keywords_en": ["agriculture", "crops", "irrigation", "agricultural production", "rural development", "forestry"],
        "keywords_ar": ["الفلاحة", "المزروعات", "السقي", "الانتاج الفلاحي", "التنمية القروية", "المياه والغابات"],
    },
    "fisheries": {
        "domains": ["peche maritime", "ressources halieutiques", "peche", "littoral"],
        "keywords_fr": ["peche", "maritime", "poisson", "ressources halieutiques", "port de peche"],
        "keywords_en": ["fisheries", "maritime", "fish", "marine resources", "fishing ports"],
        "keywords_ar": ["الصيد البحري", "الاسماك", "الموارد البحرية", "الموانئ", "الثروة السمكية"],
    },
    "museums_culture": {
        "domains": ["musees", "culture", "patrimoine", "collections"],
        "keywords_fr": ["musee", "culture", "patrimoine", "oeuvres", "collections"],
        "keywords_en": ["museum", "culture", "heritage", "artworks", "collections"],
        "keywords_ar": ["المتاحف", "الثقافة", "التراث", "الاعمال الفنية", "المجموعات"],
    },
    "statistics_demography": {
        "domains": ["statistiques", "demographie", "recensement", "indicateurs"],
        "keywords_fr": ["statistiques", "population", "emploi", "enquete", "recensement", "indicateurs"],
        "keywords_en": ["statistics", "population", "employment", "survey", "census", "indicators"],
        "keywords_ar": ["الاحصاء", "السكان", "التشغيل", "الاستبيانات", "الاحصاء العام", "المؤشرات"],
    },
    "standards_quality": {
        "domains": ["normes", "qualite", "certification", "standards"],
        "keywords_fr": ["normes", "standardisation", "qualite", "certification", "conformite"],
        "keywords_en": ["standards", "standardization", "quality", "certification", "compliance"],
        "keywords_ar": ["المعايير", "التقييس", "الجودة", "الشهادة", "المطابقة"],
    },
    "anti_corruption_governance": {
        "domains": ["probite", "corruption", "integrite", "gouvernance"],
        "keywords_fr": ["corruption", "integrite", "transparence", "probite", "gouvernance"],
        "keywords_en": ["corruption", "integrity", "transparency", "probity", "governance"],
        "keywords_ar": ["الفساد", "النزاهة", "الشفافية", "الحكامة", "الاستقامة"],
    },
    "foreign_affairs": {
        "domains": ["affaires etrangeres", "cooperation africaine", "marocains du monde", "diplomatie"],
        "keywords_fr": ["diplomatie", "etranger", "consulaire", "marocains residant a l etranger", "cooperation"],
        "keywords_en": ["foreign affairs", "diplomacy", "consular", "diaspora", "international cooperation"],
        "keywords_ar": ["الشؤون الخارجية", "الدبلوماسية", "القنصلي", "مغاربة العالم", "التعاون الدولي"],
    },
    "housing_urban_policy": {
        "domains": ["amenagement du territoire", "urbanisme", "habitat", "politique de la ville"],
        "keywords_fr": ["urbanisme", "habitat", "ville", "amenagement", "logement", "territoire"],
        "keywords_en": ["urban development", "housing", "city policy", "land planning", "territory"],
        "keywords_ar": ["اعداد التراب", "التعمير", "السكن", "سياسة المدينة", "التهيئة", "المجال"],
    },
    "public_finance": {
        "domains": ["economie", "finances", "budget", "fiscalite"],
        "keywords_fr": ["budget", "finances", "economie", "fiscalite", "depenses publiques", "recettes"],
        "keywords_en": ["budget", "finance", "economy", "tax", "public spending", "revenue"],
        "keywords_ar": ["الميزانية", "المالية", "الاقتصاد", "الضرائب", "النفقات العمومية", "المداخيل"],
    },
    "education_sports": {
        "domains": ["education", "prescolaire", "ecoles", "sports"],
        "keywords_fr": ["education", "ecole", "eleves", "prescolaire", "sport", "etablissements"],
        "keywords_en": ["education", "schools", "students", "preschool", "sports", "institutions"],
        "keywords_ar": ["التعليم", "المدارس", "التلاميذ", "التعليم الاولي", "الرياضة", "المؤسسات"],
    },
    "higher_education_research": {
        "domains": ["enseignement superieur", "universite", "recherche", "innovation"],
        "keywords_fr": ["universite", "recherche", "doctorat", "innovation", "enseignement superieur"],
        "keywords_en": ["higher education", "university", "research", "doctoral studies", "innovation"],
        "keywords_ar": ["التعليم العالي", "الجامعة", "البحث العلمي", "الدكتوراه", "الابتكار"],
    },
    "industry_trade": {
        "domains": ["industrie", "commerce", "entreprise", "export"],
        "keywords_fr": ["industrie", "commerce", "usines", "export", "import", "production"],
        "keywords_en": ["industry", "commerce", "factories", "export", "import", "production"],
        "keywords_ar": ["الصناعة", "التجارة", "المصانع", "التصدير", "الاستيراد", "الانتاج"],
    },
    "employment_skills_inclusion": {
        "domains": ["emploi", "competences", "inclusion economique", "petite entreprise"],
        "keywords_fr": ["emploi", "competences", "formation", "petite entreprise", "inclusion economique"],
        "keywords_en": ["employment", "skills", "training", "small business", "economic inclusion"],
        "keywords_ar": ["التشغيل", "المهارات", "التكوين", "المقاولة الصغيرة", "الادماج الاقتصادي"],
    },
    "infrastructure_water": {
        "domains": ["equipement", "infrastructures", "routes", "eau"],
        "keywords_fr": ["equipement", "routes", "ponts", "infrastructures", "eau", "travaux publics"],
        "keywords_en": ["infrastructure", "roads", "bridges", "water", "public works", "equipment"],
        "keywords_ar": ["التجهيز", "الطرق", "القناطر", "البنيات التحتية", "الماء", "الاشغال العمومية"],
    },
    "justice": {
        "domains": ["justice", "tribunaux", "judiciaire", "decisions de justice"],
        "keywords_fr": ["justice", "tribunal", "jugement", "decision judiciaire", "mahakim", "proces"],
        "keywords_en": ["justice", "court", "judgment", "court decision", "mahakim", "trial"],
        "keywords_ar": ["العدالة", "المحاكم", "الاحكام", "القرارات القضائية", "محاكم", "القضايا"],
    },
    "culture_youth_communication": {
        "domains": ["jeunesse", "culture", "communication", "medias"],
        "keywords_fr": ["jeunesse", "culture", "communication", "medias", "activites culturelles"],
        "keywords_en": ["youth", "culture", "communication", "media", "cultural activities"],
        "keywords_ar": ["الشباب", "الثقافة", "التواصل", "الاعلام", "الانشطة الثقافية"],
    },
    "solidarity_family": {
        "domains": ["solidarite", "insertion sociale", "famille", "protection des personnes vulnerables"],
        "keywords_fr": ["solidarite", "famille", "insertion sociale", "handicap", "protection sociale"],
        "keywords_en": ["solidarity", "family", "social inclusion", "disability", "social support"],
        "keywords_ar": ["التضامن", "الاسرة", "الادماج الاجتماعي", "الاعاقة", "الدعم الاجتماعي"],
    },
    "health_social_protection": {
        "domains": ["sante", "hopitaux", "protection sociale", "systeme de sante"],
        "keywords_fr": ["sante", "hopital", "medecin", "soins", "protection sociale", "maladie"],
        "keywords_en": ["health", "hospital", "doctor", "care", "social protection", "disease"],
        "keywords_ar": ["الصحة", "المستشفى", "الاطباء", "العلاج", "الحماية الاجتماعية", "المرض"],
    },
    "tourism_craft_social_economy": {
        "domains": ["tourisme", "artisanat", "economie sociale et solidaire"],
        "keywords_fr": ["tourisme", "artisanat", "cooperatives", "economie sociale", "voyage"],
        "keywords_en": ["tourism", "crafts", "cooperatives", "social economy", "travel"],
        "keywords_ar": ["السياحة", "الصناعة التقليدية", "التعاونيات", "الاقتصاد الاجتماعي", "السفر"],
    },
    "road_safety_transport": {
        "domains": ["securite routiere", "circulation", "accidents", "transport routier"],
        "keywords_fr": ["securite routiere", "accident", "permis", "circulation", "vehicule"],
        "keywords_en": ["road safety", "accident", "driving license", "traffic", "vehicle"],
        "keywords_ar": ["السلامة الطرقية", "حوادث السير", "رخصة السياقة", "حركة المرور", "المركبات"],
    },
    "parliament_legislation": {
        "domains": ["parlement", "lois", "legislation", "questions parlementaires"],
        "keywords_fr": ["parlement", "loi", "legislation", "depute", "seance", "question parlementaire"],
        "keywords_en": ["parliament", "law", "legislation", "member of parliament", "session"],
        "keywords_ar": ["البرلمان", "القوانين", "التشريع", "النواب", "الجلسات", "الاسئلة البرلمانية"],
    },
    "postal_services": {
        "domains": ["poste", "courrier", "colis", "services postaux"],
        "keywords_fr": ["poste", "courrier", "colis", "adresse", "service postal"],
        "keywords_en": ["post", "mail", "parcel", "address", "postal services"],
        "keywords_ar": ["البريد", "الطرود", "العنوان", "الخدمات البريدية"],
    },
    "regional_government": {
        "domains": ["region", "gouvernance regionale", "developpement territorial", "services regionaux"],
        "keywords_fr": ["region", "gouvernance regionale", "territoire", "developpement regional", "budget regional"],
        "keywords_en": ["region", "regional government", "territory", "regional development", "regional budget"],
        "keywords_ar": ["الجهة", "الحكامة الجهوية", "التراب", "التنمية الجهوية", "الميزانية الجهوية"],
    },
    "media_broadcasting": {
        "domains": ["audiovisuel", "media", "television", "production"],
        "keywords_fr": ["television", "media", "audiovisuel", "emissions", "diffusion"],
        "keywords_en": ["television", "media", "broadcast", "audiovisual", "programming"],
        "keywords_ar": ["التلفزيون", "الاعلام", "السمعي البصري", "البث", "البرامج"],
    },
}


ORG_OVERRIDES: dict[str, dict[str, Any]] = {
    "add": {
        "full_name": "Agence de Developpement du Digital",
        "aliases": ["ADD", "Agence de Developpement du Digital", "Digital Development Agency"],
        "category": "digital_transformation",
    },
    "ammc": {
        "full_name": "Autorite Marocaine du Marche des Capitaux",
        "aliases": ["AMMC", "Autorite Marocaine du Marche des Capitaux", "Moroccan Capital Market Authority"],
        "category": "capital_markets",
    },
    "anrt": {
        "full_name": "Agence Nationale de Reglementation des Telecommunications",
        "aliases": ["ANRT", "Agence Nationale de Reglementation des Telecommunications", "Telecommunications Regulator"],
        "category": "telecom",
    },
    "autorite de controle des assurances et de la prevoyance sociale (acaps)": {
        "full_name": "Autorite de Controle des Assurances et de la Prevoyance Sociale",
        "aliases": ["ACAPS", "Autorite de Controle des Assurances et de la Prevoyance Sociale", "Insurance and Social Welfare Authority"],
        "category": "insurance_social_protection",
    },
    "bank al-maghrib": {
        "full_name": "Bank Al-Maghrib",
        "aliases": ["Bank Al-Maghrib", "BAM", "Banque centrale du Maroc", "Central Bank of Morocco"],
        "category": "central_bank",
    },
    "caisse nationale de retraites et dassurances": {
        "full_name": "Caisse Nationale de Retraites et d Assurances",
        "aliases": ["CNRA", "Caisse Nationale de Retraites et d Assurances"],
        "category": "retirement_pension",
    },
    "cmr": {
        "full_name": "Caisse Marocaine des Retraites",
        "aliases": ["CMR", "Caisse Marocaine des Retraites"],
        "category": "retirement_pension",
    },
    "cnops": {
        "full_name": "Caisse Nationale des Organismes de Prevoyance Sociale",
        "aliases": ["CNOPS", "Caisse Nationale des Organismes de Prevoyance Sociale"],
        "category": "social_protection",
    },
    "cnss": {
        "full_name": "Caisse Nationale de Securite Sociale",
        "aliases": ["CNSS", "Caisse Nationale de Securite Sociale"],
        "category": "social_protection",
    },
    "hcp": {
        "full_name": "Haut Commissariat au Plan",
        "aliases": ["HCP", "Haut Commissariat au Plan", "High Commission for Planning"],
        "category": "statistics_demography",
    },
    "imanor": {
        "full_name": "Institut Marocain de Normalisation",
        "aliases": ["IMANOR", "Institut Marocain de Normalisation"],
        "category": "standards_quality",
    },
    "inpplc": {
        "full_name": "Instance Nationale de la Probite, de la Prevention et de la Lutte contre la Corruption",
        "aliases": ["INPPLC", "Instance Nationale de la Probite", "Anti-Corruption Authority"],
        "category": "anti_corruption_governance",
    },
    "maecamre": {
        "full_name": "Ministere des Affaires Etrangeres, de la Cooperation Africaine et des Marocains Resident a l Etranger",
        "aliases": ["MAECAMRE", "Ministere des Affaires Etrangeres", "Foreign Affairs Ministry"],
        "category": "foreign_affairs",
    },
    "matnuhpv": {
        "full_name": "Ministere de l Amenagement du Territoire National, de l Urbanisme, de l Habitat et de la Politique de la Ville",
        "aliases": ["MATNUHPV", "Ministere de l Urbanisme et de l Habitat"],
        "category": "housing_urban_policy",
    },
    "mef": {
        "full_name": "Ministere de l Economie et des Finances",
        "aliases": ["MEF", "Ministere de l Economie et des Finances", "Ministry of Economy and Finance"],
        "category": "public_finance",
    },
    "menps": {
        "full_name": "Ministere de l Education Nationale, du Prescolaire et des Sports",
        "aliases": ["MENPS", "Ministere de l Education Nationale", "Education Ministry"],
        "category": "education_sports",
    },
    "mesrsi": {
        "full_name": "Ministere de l Enseignement Superieur, de la Recherche Scientifique et de l Innovation",
        "aliases": ["MESRSI", "Ministere de l Enseignement Superieur", "Higher Education Ministry"],
        "category": "higher_education_research",
    },
    "mic": {
        "full_name": "Ministere de l Industrie et du Commerce",
        "aliases": ["MIC", "Ministere de l Industrie et du Commerce", "Industry and Trade Ministry"],
        "category": "industry_trade",
    },
    "miepeec": {
        "full_name": "Ministere de l Inclusion Economique, de la Petite Entreprise, de l Emploi et des Competences",
        "aliases": ["MIEPEEC", "Ministere de l Emploi et des Competences", "Employment Ministry"],
        "category": "employment_skills_inclusion",
    },
    "mj": {
        "full_name": "Ministere de la Justice",
        "aliases": ["MJ", "Ministere de la Justice", "Ministry of Justice", "Mahakim"],
        "category": "justice",
    },
    "mjcc": {
        "full_name": "Ministere de la Jeunesse, de la Culture et de la Communication",
        "aliases": ["MJCC", "Ministere de la Jeunesse, de la Culture et de la Communication"],
        "category": "culture_youth_communication",
    },
    "msisf : ministere de la solidarite, de l'insertion sociale et de la famille": {
        "full_name": "Ministere de la Solidarite, de l Insertion Sociale et de la Famille",
        "aliases": ["MSISF", "Ministere de la Solidarite, de l Insertion Sociale et de la Famille"],
        "category": "solidarity_family",
    },
    "msps": {
        "full_name": "Ministere de la Sante et de la Protection Sociale",
        "aliases": ["MSPS", "Ministere de la Sante et de la Protection Sociale", "Health Ministry"],
        "category": "health_social_protection",
    },
    "mtaess": {
        "full_name": "Ministere du Tourisme, de l Artisanat, et de l Economie Sociale et Solidaire",
        "aliases": ["MTAESS", "Ministere du Tourisme et de l Artisanat"],
        "category": "tourism_craft_social_economy",
    },
    "mtnra": {
        "full_name": "Ministere de la Transition Numerique et de la Reforme de l Administration",
        "aliases": ["MTNRA", "Ministere de la Transition Numerique", "Digital Transition Ministry"],
        "category": "digital_transformation",
    },
    "narsa": {
        "full_name": "Agence Nationale de la Securite Routiere",
        "aliases": ["NARSA", "Agence Nationale de la Securite Routiere"],
        "category": "road_safety_transport",
    },
    "regime collectif dallocation de retraite": {
        "full_name": "Regime Collectif d Allocation de Retraite",
        "aliases": ["RCAR", "Regime Collectif d Allocation de Retraite"],
        "category": "retirement_pension",
    },
    "soread 2m": {
        "full_name": "SOREAD 2M",
        "aliases": ["2M", "SOREAD 2M", "TV 2M"],
        "category": "media_broadcasting",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an enriched organizations CSV for multilingual semantic routing."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Source organizations CSV. Defaults to {DEFAULT_INPUT_PATH}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Destination enriched CSV. Defaults to {DEFAULT_OUTPUT_PATH}.",
    )
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def simplify_text(value: str) -> str:
    normalized = normalize_text(value).lower()
    normalized = (
        unicodedata.normalize("NFKD", normalized)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    normalized = normalized.replace("’", "'")
    return normalized


def join_unique(items: list[str]) -> str:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        clean_item = normalize_text(item)
        if not clean_item:
            continue
        key = clean_item.casefold()
        if key in seen:
            continue
        seen.add(key)
        output.append(clean_item)
    return "|".join(output)


def infer_category(organization: str, description: str) -> str:
    lookup = simplify_text(organization)
    combined = f"{lookup} {simplify_text(description)}"

    if lookup in ORG_OVERRIDES:
        return str(ORG_OVERRIDES[lookup]["category"])
    if "justice" in combined or lookup == "mj":
        return "justice"
    if "transition numerique" in combined or "digital" in combined:
        return "digital_transformation"
    if "bassin hydraulique" in combined:
        return "water_resources"
    if "agence urbaine" in combined:
        return "urban_planning"
    if "marche des capitaux" in combined:
        return "capital_markets"
    if "telecommunications" in combined:
        return "telecom"
    if "archives" in combined:
        return "archives_heritage"
    if "assurances" in combined and "prevoyance" in combined:
        return "insurance_social_protection"
    if "bank al-maghrib" in combined:
        return "central_bank"
    if "retraites" in combined or "allocation de retraite" in combined or lookup in {"cmr"}:
        return "retirement_pension"
    if "investissement" in combined:
        return "investment_regional"
    if "commerce" in combined and "industrie" in combined and "chambre" in combined:
        return "commerce_industry_regional"
    if lookup in {"cnops", "cnss"} or "prevoyance sociale" in combined or "securite sociale" in combined:
        return "social_protection"
    if "commune" in combined or "collectivites territoriales" in combined:
        return "local_government"
    if "agriculture" in combined:
        return "agriculture"
    if "peche maritime" in combined:
        return "fisheries"
    if "musees" in combined:
        return "museums_culture"
    if lookup == "hcp" or "commissariat au plan" in combined:
        return "statistics_demography"
    if lookup == "imanor" or "normalisation" in combined:
        return "standards_quality"
    if lookup == "inpplc" or "corruption" in combined or "probite" in combined:
        return "anti_corruption_governance"
    if lookup == "maecamre" or "affaires etrangeres" in combined:
        return "foreign_affairs"
    if lookup == "matnuhpv" or "politique de la ville" in combined or "amenagement du territoire" in combined:
        return "housing_urban_policy"
    if lookup == "mef" or "economie et des finances" in combined:
        return "public_finance"
    if lookup == "menps" or "prescolaire" in combined or "sports" in combined:
        return "education_sports"
    if lookup == "mesrsi" or "enseignement superieur" in combined or "recherche scientifique" in combined:
        return "higher_education_research"
    if lookup == "mic" or "industrie et du commerce" in combined:
        return "industry_trade"
    if lookup == "miepeec" or "petite entreprise" in combined or "competences" in combined:
        return "employment_skills_inclusion"
    if "equipement" in combined and "eau" in combined:
        return "infrastructure_water"
    if lookup == "mjcc" or "jeunesse" in combined or "communication" in combined:
        return "culture_youth_communication"
    if lookup.startswith("msisf") or "insertion sociale" in combined:
        return "solidarity_family"
    if lookup == "msps" or "sante" in combined:
        return "health_social_protection"
    if lookup == "mtaess" or "tourisme" in combined or "artisanat" in combined:
        return "tourism_craft_social_economy"
    if lookup == "narsa" or "securite routiere" in combined:
        return "road_safety_transport"
    if "parlement" in combined:
        return "parliament_legislation"
    if "poste" in combined:
        return "postal_services"
    if lookup.startswith("region "):
        return "regional_government"
    if "soread" in combined or "2m" in combined:
        return "media_broadcasting"
    return "local_government"


def infer_aliases(organization: str, full_name: str, category: str) -> list[str]:
    aliases = [organization]
    if full_name.casefold() != organization.casefold():
        aliases.append(full_name)

    if "(" in organization and ")" in organization:
        aliases.extend(re.findall(r"\(([^)]+)\)", organization))

    if category == "justice":
        aliases.extend(["Ministry of Justice", "Mahakim"])
    elif category == "statistics_demography":
        aliases.append("High Commission for Planning")
    elif category == "digital_transformation":
        aliases.append("Digital administration")

    return aliases


def build_domains(organization: str, profile: dict[str, list[str] | str]) -> list[str]:
    domains = list(profile["domains"])  # type: ignore[index]
    domains.append(organization)
    return domains


def build_profile_text(
    full_name: str,
    aliases: str,
    category: str,
    domains: str,
    keywords_fr: str,
    keywords_en: str,
    keywords_ar: str,
    source_description: str,
    package_count: int,
) -> str:
    return normalize_text(
        (
            f"{full_name}. "
            f"Category: {category}. "
            f"Aliases: {aliases}. "
            f"Domaines: {domains}. "
            f"Requetes FR: {keywords_fr}. "
            f"Related EN: {keywords_en}. "
            f"طلبات AR: {keywords_ar}. "
            f"Description source: {source_description}. "
            f"Open data packages: {package_count}."
        )
    )


def build_row(source_row: dict[str, str]) -> dict[str, Any]:
    organization = normalize_text(source_row.get("organization"))
    source_description = normalize_text(source_row.get("description"))
    package_count = int(source_row.get("package_count") or 0)
    organization_slug = normalize_text(source_row.get("organization_slug"))
    key = simplify_text(organization)

    override = ORG_OVERRIDES.get(key, {})
    full_name = normalize_text(override.get("full_name") or organization)
    category = normalize_text(override.get("category") or infer_category(organization, source_description))
    category_profile = CATEGORY_PROFILES[category]

    base_aliases = override.get("aliases") or infer_aliases(organization, full_name, category)
    aliases = join_unique(list(base_aliases))
    domains = join_unique(build_domains(organization, category_profile))
    keywords_fr = join_unique(list(category_profile["keywords_fr"]))  # type: ignore[index]
    keywords_en = join_unique(list(category_profile["keywords_en"]))  # type: ignore[index]
    keywords_ar = join_unique(list(category_profile["keywords_ar"]))  # type: ignore[index]

    profile_text = build_profile_text(
        full_name=full_name,
        aliases=aliases.replace("|", ", "),
        category=category,
        domains=domains.replace("|", ", "),
        keywords_fr=keywords_fr.replace("|", ", "),
        keywords_en=keywords_en.replace("|", ", "),
        keywords_ar=keywords_ar.replace("|", ", "),
        source_description=source_description,
        package_count=package_count,
    )

    return {
        "organization": organization,
        "organization_slug": organization_slug,
        "full_name": full_name,
        "aliases": aliases,
        "category": category,
        "domains": domains,
        "keywords_fr": keywords_fr,
        "keywords_en": keywords_en,
        "keywords_ar": keywords_ar,
        "source_description": source_description,
        "profile_text": profile_text,
        "package_count": package_count,
    }


def read_rows(input_path: Path) -> list[dict[str, str]]:
    with input_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def write_rows(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    source_rows = read_rows(args.input)
    output_rows = [build_row(row) for row in source_rows]
    write_rows(output_rows, args.output)
    print(f"Built {len(output_rows)} enriched organization profiles in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
