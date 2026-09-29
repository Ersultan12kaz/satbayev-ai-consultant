"""Tiny synthetic corpus used ONLY for automated tests (tests/) and a quick smoke run.

The texts are short paraphrased examples in the style of university regulations;
they are NOT the official documents.  The real corpus is downloaded by
scripts/01_download_documents.py.
"""
import json
from pathlib import Path

DOCS = {
    ("ru", "Академическая политика 2022"): [
        "1. Общие положения\nАкадемическая политика определяет правила организации учебного процесса по кредитной технологии обучения. Обучающийся формирует индивидуальный учебный план с помощью эдвайзера.",
        "2. Академический отпуск\nАкадемический отпуск предоставляется обучающемуся по медицинским показаниям, в связи с призывом на воинскую службу, рождением ребенка. Для оформления академического отпуска студент подает заявление на имя руководителя и прилагает заключение врачебно-консультационной комиссии или повестку. Срок академического отпуска по болезни составляет от шести месяцев до одного года.",
        "3. Отчисление и восстановление\nСтудент может быть отчислен за академическую неуспеваемость, нарушение правил внутреннего распорядка или по собственному желанию. Восстановление производится в период летних и зимних каникул при условии выполнения требований учебного плана.",
        "4. Перевод\nПеревод с одной образовательной программы на другую осуществляется в период каникул. Академическая разница в дисциплинах ликвидируется в летнем семестре. Кредиты перезачитываются на основании транскрипта.",
    ],
    ("ru", "Положение по организации и проведению экзаменационной сессии 2023"): [
        "1. Экзаменационная сессия\nК экзамену допускаются обучающиеся, набравшие рейтинг допуска не менее 50 баллов. Экзамен проводится в письменной, устной или тестовой форме.",
        "2. Апелляция\nОбучающийся, не согласный с результатом экзамена, подает заявление на апелляцию не позднее следующего рабочего дня после объявления результатов. Апелляционная комиссия рассматривает заявление в течение трех рабочих дней.",
        "3. Оценка FX\nОценка FX означает возможность пересдачи итогового контроля без повторного прохождения дисциплины. Пересдача экзамена с положительной оценки с целью ее повышения не допускается.",
    ],
    ("ru", "Правила о стипендиальном обеспечении обучающихся 2022"): [
        "1. Государственная стипендия\nГосударственная стипендия назначается обучающимся по государственному образовательному заказу по итогам экзаменационной сессии при условии сдачи всех экзаменов на оценки «отлично» и «хорошо». Стипендия выплачивается ежемесячно.",
        "2. Материальная поддержка\nОбучающимся из числа сирот и лиц с инвалидностью оказывается социальная поддержка. Скидки на обучение предоставляются отличникам и победителям олимпиад.",
    ],
    ("ru", "Правила проживания в общежитиях 2023"): [
        "1. Заселение\nДля заселения в общежитие студент предоставляет направление, удостоверение личности, медицинскую справку и договор найма. Места в общежитии распределяются комиссией.",
        "2. Правила проживания\nПроживающие обязаны соблюдать тишину после 23:00, бережно относиться к имуществу и соблюдать правила пожарной безопасности. Посещение общежития гостями разрешается до 21:00 при предъявлении документа.",
        "3. Выселение\nСтудент может быть выселен из общежития за систематическое нарушение правил проживания, распитие спиртных напитков и неоплату проживания.",
    ],
    ("ru", "ДП КазНИТУ 709 Профессиональная практика (Бакалавриат)"): [
        "1. Виды практики\nПрофессиональная практика включает учебную, производственную и преддипломную практику. Базы практики определяются договорами с организациями.",
        "2. Отчетность\nПо итогам практики студент сдает отчет, дневник практики и отзыв руководителя от организации. Практика оценивается по результатам защиты отчета.",
    ],
    ("ru", "Положение о проведении итоговой аттестации (Undergraduate Thesis) 2021"): [
        "1. Итоговая аттестация\nИтоговая аттестация проводится в форме защиты дипломной работы или проекта. К итоговой аттестации допускаются обучающиеся, полностью выполнившие учебный план.",
        "2. Проверка на плагиат\nДипломная работа проверяется на плагиат. Работа с коэффициентом оригинальности ниже установленного порога к защите не допускается.",
    ],
    ("kk", "Академиялық саясат 2022"): [
        "1. Жалпы ережелер\nАкадемиялық саясат кредиттік оқыту технологиясы бойынша оқу процесін ұйымдастыру қағидаларын айқындайды. Білім алушы эдвайзердің көмегімен жеке оқу жоспарын құрады.",
        "2. Академиялық демалыс\nАкадемиялық демалыс білім алушыға медициналық көрсеткіштер бойынша, әскери қызметке шақырылуына байланысты, бала тууына байланысты беріледі. Академиялық демалысты рәсімдеу үшін студент басшының атына өтініш береді және дәрігерлік-консультациялық комиссияның қорытындысын немесе шақыру қағазын қоса береді.",
        "3. Оқудан шығару және қайта қабылдау\nСтудент академиялық үлгермеушілігі, ішкі тәртіп ережелерін бұзғаны үшін немесе өз қалауы бойынша оқудан шығарылуы мүмкін. Қайта қабылдау жазғы және қысқы демалыс кезеңінде жүргізіледі.",
    ],
    ("kk", "Қ.И.Сәтбаев атындағы ҚазҰТЗУ жатақханаларында тұру Қағидалары 2023"): [
        "1. Қоныстану\nЖатақханаға қоныстану үшін студент жолдама, жеке куәлік, медициналық анықтама және жалдау шартын ұсынады. Жатақханадағы орындарды комиссия бөледі.",
        "2. Тұру қағидалары\nТұрғындар сағат 23:00-ден кейін тыныштықты сақтауға, мүлікке ұқыпты қарауға және өрт қауіпсіздігі қағидаларын сақтауға міндетті. Қонақтарға жатақханаға сағат 21:00-ге дейін құжатын көрсеткен жағдайда кіруге рұқсат етіледі.",
    ],
    ("kk", "Білім алушыларын стипендиямен қамтамасыз ету қағидалары"): [
        "1. Мемлекеттік шәкіртақы\nМемлекеттік шәкіртақы мемлекеттік білім беру тапсырысы бойынша оқитын білім алушыларға емтихан сессиясының қорытындысы бойынша барлық емтихандарды «өте жақсы» және «жақсы» бағаларына тапсырған жағдайда тағайындалады.",
        "2. Материалдық қолдау\nЖетім және мүгедектігі бар білім алушыларға әлеуметтік қолдау көрсетіледі. Оқу ақысына жеңілдіктер үздік студенттерге беріледі.",
    ],
    ("kk", "Емтихан сессиясын ұйымдастыру және өткізу туралы ереже"): [
        "1. Емтихан сессиясы\nЕмтиханға кемінде 50 балл жіберу рейтингін жинаған білім алушылар жіберіледі. Емтихан жазбаша, ауызша немесе тест түрінде өткізіледі.",
        "2. Апелляция\nЕмтихан нәтижесімен келіспеген білім алушы нәтижелер жарияланғаннан кейінгі келесі жұмыс күнінен кешіктірмей апелляцияға өтініш береді.",
    ],
    ("en", "Academic Policy 2022"): [
        "1. General provisions\nThe academic policy defines the rules of organizing the educational process under the credit technology of education. The student forms an individual study plan with the help of an adviser.",
        "2. Academic leave\nAcademic leave is granted to a student for medical reasons, in connection with conscription for military service, or the birth of a child. To apply for academic leave the student submits an application addressed to the head and attaches the conclusion of the medical advisory commission or the summons.",
        "3. Transfer\nTransfer from one educational program to another is carried out during vacations. The academic difference in disciplines is eliminated in the summer semester.",
    ],
    ("en", "Regulation for the allocation of the places in the dormitories 2023"): [
        "1. Allocation of places\nPlaces in the dormitories are allocated by the commission. Priority is given to orphans, students with disabilities and students from large families.",
        "2. Moving in\nTo move into the dormitory a student provides a referral, an identity card, a medical certificate and a rental agreement.",
    ],
    ("en", "Regulation on the organization and conduct of the examination session 2023"): [
        "1. Examination session\nStudents who have an admission rating of at least 50 points are admitted to the exam. The exam is held in written, oral or test form.",
        "2. Appeal\nA student who disagrees with the exam result submits an appeal application no later than the next working day after the results are announced. The FX grade means the student can retake the final control without repeating the course.",
    ],
    ("en", "Regulations on the final attestation (Undergraduate Thesis) 2021"): [
        "1. Final attestation\nThe final attestation is conducted in the form of a thesis defense. Students who have fully completed the curriculum are admitted to the final attestation.",
        "2. Plagiarism\nThe thesis is checked for plagiarism. A thesis with an originality below the established threshold is not admitted to the defense.",
    ],
}


def build(out_dir: Path) -> Path:
    raw = out_dir / "raw"
    manifest = []
    for i, ((lang, title), pages) in enumerate(DOCS.items(), start=1):
        p = raw / lang / f"{i}_{title[:60]}.txt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n\n".join(pages), encoding="utf-8")
        manifest.append({"doc_id": f"{lang}_{i}", "doc_num": str(i), "lang": lang, "category": "test",
                         "title": title, "file_name": p.name, "url": f"https://example.org/{i}",
                         "path": str(p.relative_to(raw)).replace("\\", "/"), "size": p.stat().st_size,
                         "pages": pages})
    with open(raw / "manifest.jsonl", "w", encoding="utf-8") as f:
        for m in manifest:
            f.write(json.dumps({k: v for k, v in m.items() if k != "pages"}, ensure_ascii=False) + "\n")
    return raw


def as_chunks():
    """Chunks (one per page) for fast unit tests."""
    out = []
    for i, ((lang, title), pages) in enumerate(DOCS.items(), start=1):
        for pno, text in enumerate(pages, start=1):
            head, _, body = text.partition("\n")
            out.append({"chunk_id": f"{lang}_{i}_{pno:04d}", "doc_id": f"{lang}_{i}", "title": title, "lang": lang,
                        "category": "test", "url": f"https://example.org/{i}", "page": pno, "section": head,
                        "text": body, "n_words": len(body.split())})
    return out
