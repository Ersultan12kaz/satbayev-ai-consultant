"""Source of the evaluation set (run it to regenerate the JSONL files).

10 categories x 10 questions x 3 languages = 300 parallel questions (RU / KZ / EN).
For every question the expected official source is given as regular expressions
over document titles (any language version of the document counts).

    python data/eval/eval_questions_source.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

SOURCES = {
    "admission": [r"при[её]м", r"қабылда", r"admission", r"702", r"контингент"],
    "education": [r"академическ\w* политик", r"академиялық саясат", r"academic policy", r"кредитн\w* технолог",
                  r"кредиттік", r"credit\s+technology", r"705", r"учебный процесс", r"оқу процес", r"educational process"],
    "exams": [r"экзаменационн", r"емтихан", r"examination", r"706", r"оценка знаний", r"білімді бағалау",
              r"knowledge assessment", r"апелляц", r"appeal", r"академическ\w* политик", r"академиялық саясат",
              r"academic policy", r"кредит", r"credit", r"онлайн-экзамен", r"экзаменатор"],
    "scholarship": [r"стипенди", r"шәкіртақы", r"scholarship", r"грант", r"grant", r"скидк", r"жеңілдік",
                    r"discount", r"715", r"социальн\w* поддерж", r"әлеуметтік қолда", r"social support", r"адилов"],
    "dormitory": [r"общежит", r"жатақхана", r"dormitor", r"715", r"социальн\w* поддерж", r"social support",
                  r"әлеуметтік қолда"],
    "academic_leave": [r"академическ\w* политик", r"академиялық саясат", r"academic policy", r"кредит", r"credit",
                       r"702", r"контингент", r"705", r"учебный процесс", r"оқу процес"],
    "transfer": [r"702", r"контингент", r"submission of students", r"академическ\w* политик", r"академиялық саясат",
                 r"academic policy", r"кредит", r"credit", r"718", r"мобильн", r"ұтқырлық", r"mobility",
                 r"двудипломн", r"қос дипломды", r"double.degree"],
    "internship": [r"709", r"практик", r"practice", r"internship", r"дуальн", r"дуальды", r"dual", r"710",
                   r"трудоустр", r"жұмысқа орналастыру", r"employment"],
    "graduation": [r"итогов\w* аттестац", r"қорытынды аттестат", r"final attestation", r"плагиат", r"plagiarism",
                   r"диплом", r"diploma", r"ects", r"ests", r"дипломирован"],
    "student_services": [r"эдвайзер", r"adviser", r"advis", r"самоуправлен", r"өзін-өзі басқару", r"self-govern",
                         r"этическ", r"этикалық", r"ethic", r"дистанцион", r"қашықтықтан", r"distance", r"онлайн",
                         r"online", r"библиотек", r"кітапхана", r"library", r"minor", r"713", r"воспитательн",
                         r"тәрбие", r"extracurricular", r"инклюзив", r"инклюзивті", r"inclusive", r"учебной системы",
                         r"иностранн", r"шетелдік", r"foreign", r"718", r"мобильн", r"mobility", r"ұтқырлық"],
}

Q = {
    "admission": [
        ("Какие документы нужны для поступления на бакалавриат?", "Бакалавриатқа түсу үшін қандай құжаттар қажет?", "What documents are required for admission to the bachelor's program?"),
        ("Какой минимальный балл ЕНТ нужен для поступления?", "Түсу үшін ҰБТ-ның ең төменгі балы қандай?", "What is the minimum UNT score required for admission?"),
        ("Когда начинается прием документов в университет?", "Университетке құжаттарды қабылдау қашан басталады?", "When does the university start accepting applications?"),
        ("Как поступить в магистратуру Satbayev University?", "Satbayev University магистратурасына қалай түсуге болады?", "How can I enter the master's program at Satbayev University?"),
        ("Какие вступительные экзамены сдают в докторантуру?", "Докторантураға қандай түсу емтихандары тапсырылады?", "What entrance exams are required for doctoral studies?"),
        ("Как поступают иностранные граждане?", "Шетел азаматтары қалай оқуға түседі?", "How do foreign citizens apply for admission?"),
        ("Нужен ли сертификат по английскому языку для магистратуры?", "Магистратураға ағылшын тілі сертификаты қажет пе?", "Is an English language certificate required for the master's program?"),
        ("Как зачисляют на платное обучение?", "Ақылы оқуға қалай қабылданады?", "How are students enrolled on a paid basis?"),
        ("Что такое приемная комиссия и чем она занимается?", "Қабылдау комиссиясы деген не және ол немен айналысады?", "What does the admission committee do?"),
        ("Какие правила приема на программы MBA?", "MBA бағдарламаларына қабылдау қағидалары қандай?", "What are the admission rules for MBA programs?"),
    ],
    "education": [
        ("Что такое кредитная технология обучения?", "Оқытудың кредиттік технологиясы деген не?", "What is the credit technology of education?"),
        ("Как рассчитывается GPA?", "GPA қалай есептеледі?", "How is GPA calculated?"),
        ("Что такое индивидуальный учебный план студента?", "Студенттің жеке оқу жоспары деген не?", "What is a student's individual study plan?"),
        ("Сколько кредитов нужно освоить за учебный год?", "Бір оқу жылында қанша кредит меңгеру керек?", "How many credits must be completed per academic year?"),
        ("Как происходит регистрация на дисциплины?", "Пәндерге тіркелу қалай жүргізіледі?", "How does registration for courses work?"),
        ("Что такое летний семестр?", "Жазғы семестр деген не?", "What is the summer semester?"),
        ("Какая система оценок используется в университете?", "Университетте қандай бағалау жүйесі қолданылады?", "What grading system is used at the university?"),
        ("Какие требования к посещаемости занятий?", "Сабаққа қатысуға қандай талаптар қойылады?", "What are the class attendance requirements?"),
        ("Что такое академический календарь?", "Академиялық күнтізбе деген не?", "What is the academic calendar?"),
        ("Что такое элективные дисциплины?", "Элективті пәндер деген не?", "What are elective courses?"),
    ],
    "exams": [
        ("Как проходит экзаменационная сессия?", "Емтихан сессиясы қалай өтеді?", "How is the examination session organized?"),
        ("Что такое рейтинг допуска к экзамену?", "Емтиханға жіберу рейтингі деген не?", "What is the admission rating for the exam?"),
        ("Как подать апелляцию на оценку экзамена?", "Емтихан бағасына апелляцияны қалай беруге болады?", "How can I appeal an exam grade?"),
        ("Можно ли пересдать экзамен с положительной оценкой?", "Оң бағамен тапсырылған емтиханды қайта тапсыруға бола ма?", "Can I retake an exam that I passed with a positive grade?"),
        ("Что означает оценка FX?", "FX бағасы нені білдіреді?", "What does the FX grade mean?"),
        ("Что делать, если я пропустил экзамен по уважительной причине?", "Емтиханды дәлелді себеппен өткізіп алсам не істеу керек?", "What should I do if I missed an exam for a valid reason?"),
        ("Как проводится онлайн-экзамен?", "Онлайн емтихан қалай өткізіледі?", "How is an online exam conducted?"),
        ("Как ликвидировать академическую задолженность?", "Академиялық берешекті қалай жоюға болады?", "How can I eliminate academic debt?"),
        ("Сколько длится экзамен?", "Емтихан қанша уақытқа созылады?", "How long does an exam last?"),
        ("Что будет, если меня поймают на списывании на экзамене?", "Емтиханда көшіріп жазып ұсталсам не болады?", "What happens if I am caught cheating during an exam?"),
    ],
    "scholarship": [
        ("Кто имеет право на получение стипендии?", "Шәкіртақы алуға кімнің құқығы бар?", "Who is eligible for a scholarship?"),
        ("Какой средний балл нужен для стипендии?", "Шәкіртақы алу үшін қандай орташа балл қажет?", "What average grade is required to receive a scholarship?"),
        ("Как назначается государственная стипендия?", "Мемлекеттік шәкіртақы қалай тағайындалады?", "How is the state scholarship awarded?"),
        ("Какие формы материальной поддержки есть для студентов?", "Студенттерге материалдық қолдаудың қандай түрлері бар?", "What forms of financial support are available to students?"),
        ("Как получить скидку на обучение?", "Оқу ақысына жеңілдікті қалай алуға болады?", "How can I get a tuition discount?"),
        ("Что такое именная стипендия?", "Атаулы шәкіртақы деген не?", "What is a named scholarship?"),
        ("Сохраняется ли стипендия во время академического отпуска?", "Академиялық демалыс кезінде шәкіртақы сақтала ма?", "Is the scholarship kept during academic leave?"),
        ("Как получить грант университета?", "Университет грантын қалай алуға болады?", "How can I get a university grant?"),
        ("Какая социальная поддержка есть для студентов-сирот?", "Жетім студенттерге қандай әлеуметтік қолдау бар?", "What social support is available for orphan students?"),
        ("Когда выплачивается стипендия?", "Шәкіртақы қашан төленеді?", "When is the scholarship paid?"),
    ],
    "dormitory": [
        ("Как получить место в общежитии?", "Жатақханадан орынды қалай алуға болады?", "How can I get a place in the dormitory?"),
        ("Какие документы нужны для заселения в общежитие?", "Жатақханаға қоныстану үшін қандай құжаттар қажет?", "What documents are required to move into the dormitory?"),
        ("Кто имеет приоритет при распределении мест в общежитии?", "Жатақханада орын бөлуде кімнің басымдығы бар?", "Who has priority in the allocation of dormitory places?"),
        ("Какие правила проживания в общежитии?", "Жатақханада тұру қағидалары қандай?", "What are the rules of living in the dormitory?"),
        ("Можно ли приводить гостей в общежитие?", "Жатақханаға қонақ әкелуге бола ма?", "Can I bring guests to the dormitory?"),
        ("За что могут выселить из общежития?", "Жатақханадан не үшін шығарып жіберуі мүмкін?", "For what reasons can a student be evicted from the dormitory?"),
        ("Какие обязанности у проживающих в общежитии?", "Жатақханада тұратындардың міндеттері қандай?", "What are the duties of dormitory residents?"),
        ("Какие меры пожарной безопасности действуют в общежитии?", "Жатақханада қандай өрт қауіпсіздігі шаралары қолданылады?", "What fire safety measures apply in the dormitory?"),
        ("Как оплачивается проживание в общежитии?", "Жатақханада тұру ақысы қалай төленеді?", "How is dormitory accommodation paid for?"),
        ("Кто распределяет места в общежитиях?", "Жатақханалардағы орындарды кім бөледі?", "Who allocates places in the dormitories?"),
    ],
    "academic_leave": [
        ("Как оформить академический отпуск?", "Академиялық демалысты қалай рәсімдеуге болады?", "How do I apply for academic leave?"),
        ("Какие документы нужны для академического отпуска?", "Академиялық демалыс үшін қандай құжаттар қажет?", "What documents are required for academic leave?"),
        ("По каким причинам предоставляется академический отпуск?", "Академиялық демалыс қандай себептермен беріледі?", "For what reasons is academic leave granted?"),
        ("На какой срок дается академический отпуск?", "Академиялық демалыс қанша мерзімге беріледі?", "For how long is academic leave granted?"),
        ("Как выйти из академического отпуска?", "Академиялық демалыстан қалай шығуға болады?", "How do I return from academic leave?"),
        ("Можно ли взять академический отпуск по болезни?", "Ауруға байланысты академиялық демалыс алуға бола ма?", "Can I take academic leave due to illness?"),
        ("Дается ли академический отпуск в связи с призывом в армию?", "Әскерге шақырылуына байланысты академиялық демалыс беріле ме?", "Is academic leave granted for military service?"),
        ("Как восстановиться в университете после отчисления?", "Оқудан шығарылғаннан кейін университетке қалай қайта қабылдануға болады?", "How can I be reinstated at the university after expulsion?"),
        ("За что студента могут отчислить?", "Студентті не үшін оқудан шығаруы мүмкін?", "For what reasons can a student be expelled?"),
        ("Что такое повторное обучение?", "Қайта оқу деген не?", "What is repeated study of a course?"),
    ],
    "transfer": [
        ("Как перевестись на другую образовательную программу?", "Басқа білім беру бағдарламасына қалай ауысуға болады?", "How can I transfer to another educational program?"),
        ("Как перевестись из другого вуза в Satbayev University?", "Басқа ЖОО-дан Satbayev University-ге қалай ауысуға болады?", "How can I transfer from another university to Satbayev University?"),
        ("Когда можно подать заявление на перевод?", "Ауысуға өтінішті қашан беруге болады?", "When can I submit an application for transfer?"),
        ("Как перезачитываются кредиты при переводе?", "Ауысу кезінде кредиттер қалай қайта есептеледі?", "How are credits recognized when transferring?"),
        ("Что такое академическая разница?", "Академиялық айырмашылық деген не?", "What is the academic difference?"),
        ("Как перевестись с платного обучения на грант?", "Ақылы оқудан грантқа қалай ауысуға болады?", "How can I transfer from paid tuition to a state grant?"),
        ("Что такое академическая мобильность?", "Академиялық ұтқырлық деген не?", "What is academic mobility?"),
        ("Как поехать учиться по обмену за границу?", "Шетелге алмасу бағдарламасымен оқуға қалай баруға болады?", "How can I study abroad on an exchange program?"),
        ("Как признаются результаты обучения, полученные в другом вузе?", "Басқа ЖОО-да алынған оқу нәтижелері қалай танылады?", "How are learning outcomes obtained at another university recognized?"),
        ("Что такое двудипломная программа?", "Қос дипломды бағдарлама деген не?", "What is a double degree program?"),
    ],
    "internship": [
        ("Какие виды практики есть у студентов бакалавриата?", "Бакалавриат студенттерінде практиканың қандай түрлері бар?", "What types of internship do bachelor students have?"),
        ("Как организуется профессиональная практика?", "Кәсіптік практика қалай ұйымдастырылады?", "How is professional practice organized?"),
        ("Как выбрать базу практики?", "Практика базасын қалай таңдауға болады?", "How do I choose a place for my internship?"),
        ("Какие документы нужно сдать по итогам практики?", "Практика қорытындысы бойынша қандай құжаттар тапсыру керек?", "What documents must be submitted after the internship?"),
        ("Как оценивается практика?", "Практика қалай бағаланады?", "How is the internship assessed?"),
        ("Кто является руководителем практики?", "Практика жетекшісі кім?", "Who supervises the internship?"),
        ("Что такое дуальное обучение?", "Дуальды оқыту деген не?", "What is dual education?"),
        ("Как университет помогает с трудоустройством выпускников?", "Университет түлектерді жұмысқа орналастыруға қалай көмектеседі?", "How does the university help graduates find employment?"),
        ("Сколько длится производственная практика?", "Өндірістік практика қанша уақытқа созылады?", "How long does the industrial internship last?"),
        ("Что делать, если я не прошел практику?", "Практиканы өтпей қалсам не істеу керек?", "What should I do if I failed the internship?"),
    ],
    "graduation": [
        ("Как проходит итоговая аттестация?", "Қорытынды аттестаттау қалай өтеді?", "How is the final attestation conducted?"),
        ("Какие требования к дипломной работе?", "Дипломдық жұмысқа қандай талаптар қойылады?", "What are the requirements for the graduation thesis?"),
        ("Как проверяют дипломную работу на плагиат?", "Дипломдық жұмыс плагиатқа қалай тексеріледі?", "How is the thesis checked for plagiarism?"),
        ("Какой допустимый процент оригинальности работы?", "Жұмыстың түпнұсқалығының рұқсат етілген пайызы қандай?", "What is the acceptable percentage of originality of the thesis?"),
        ("Как проходит защита дипломного проекта?", "Дипломдық жобаны қорғау қалай өтеді?", "How is the thesis defense conducted?"),
        ("Кто допускается к итоговой аттестации?", "Қорытынды аттестаттауға кім жіберіледі?", "Who is admitted to the final attestation?"),
        ("Что такое приложение к диплому ECTS?", "ECTS дипломына қосымша деген не?", "What is the ECTS diploma supplement?"),
        ("Как получить диплом с отличием?", "Үздік дипломды қалай алуға болады?", "How can I get a diploma with honors?"),
        ("Что такое диплом собственного образца?", "Өз үлгісіндегі диплом деген не?", "What is a diploma of the university's own sample?"),
        ("Что делать, если дипломная работа не прошла проверку на плагиат?", "Дипломдық жұмыс плагиат тексеруінен өтпесе не істеу керек?", "What happens if my thesis fails the plagiarism check?"),
    ],
    "student_services": [
        ("Кто такой эдвайзер и чем он помогает студенту?", "Эдвайзер кім және ол студентке қалай көмектеседі?", "Who is an adviser and how does the adviser help students?"),
        ("Как работает студенческое самоуправление?", "Студенттік өзін-өзі басқару қалай жұмыс істейді?", "How does student self-government work?"),
        ("Что написано в этическом кодексе студента?", "Білім алушылардың этикалық кодексінде не жазылған?", "What does the student code of ethics say?"),
        ("Как организовано дистанционное обучение?", "Қашықтықтан оқыту қалай ұйымдастырылған?", "How is distance learning organized?"),
        ("Как пользоваться электронной библиотекой?", "Электрондық кітапхананы қалай пайдалануға болады?", "How do I use the electronic library?"),
        ("Что такое программа Minor?", "Minor бағдарламасы деген не?", "What is the Minor program?"),
        ("Как организовано инклюзивное обучение для студентов с ограниченными возможностями?", "Мүмкіндігі шектеулі студенттер үшін инклюзивті оқыту қалай ұйымдастырылған?", "How is inclusive education organized for students with disabilities?"),
        ("Какие внеучебные мероприятия проводит университет?", "Университет қандай сабақтан тыс іс-шаралар өткізеді?", "What extracurricular activities does the university organize?"),
        ("Как иностранному студенту оформить пребывание?", "Шетелдік студент тұруды қалай рәсімдейді?", "How does a foreign student register their stay?"),
        ("Как работать в учебной системе университета?", "Университеттің оқу жүйесінде қалай жұмыс істеуге болады?", "How do I use the university's learning system?"),
    ],
}

OUT_OF_SCOPE = [
    ("Кто выиграл чемпионат мира по футболу?", "Футболдан әлем чемпионатын кім жеңді?", "Who won the football World Cup?"),
    ("Какая погода будет завтра в Алматы?", "Ертең Алматыда ауа райы қандай болады?", "What will the weather be like in Almaty tomorrow?"),
    ("Как приготовить плов?", "Палауды қалай пісіруге болады?", "How do I cook pilaf?"),
    ("Какой сейчас курс доллара?", "Қазір доллар бағамы қандай?", "What is the current dollar exchange rate?"),
    ("Посоветуй хороший фильм на вечер", "Кешке жақсы фильм ұсыншы", "Recommend a good movie for tonight"),
    ("Кто такой Илон Маск?", "Илон Маск кім?", "Who is Elon Musk?"),
    ("Какой смартфон лучше купить?", "Қай смартфонды сатып алған дұрыс?", "Which smartphone should I buy?"),
    ("Расскажи смешной анекдот", "Күлкілі анекдот айтып берші", "Tell me a funny joke"),
    ("Сколько лететь до Марса?", "Марсқа дейін қанша уақыт ұшу керек?", "How long does it take to fly to Mars?"),
    ("Как быстро похудеть?", "Қалай тез арықтауға болады?", "How can I lose weight fast?"),
]

UNANSWERABLE = [
    ("Сколько стоит обед в студенческой столовой?", "Студенттік асханада түскі ас қанша тұрады?", "How much does lunch cost in the student canteen?"),
    ("Какой пароль от Wi-Fi в главном корпусе?", "Бас ғимараттағы Wi-Fi құпия сөзі қандай?", "What is the Wi-Fi password in the main building?"),
    ("Во сколько открывается спортзал в субботу?", "Сенбі күні спорт зал сағат нешеде ашылады?", "What time does the gym open on Saturday?"),
    ("Как зовут декана моего факультета?", "Менің факультетімнің деканының аты кім?", "What is the name of my faculty's dean?"),
    ("Какой номер телефона у охраны общежития №3?", "№3 жатақхана күзетшісінің телефон нөмірі қандай?", "What is the phone number of dormitory No. 3 security?"),
]

# the official freshman guide (Бірінші курстарға арналған анықтама-кітапша) covers every student topic
for _cat in SOURCES:
    SOURCES[_cat] = SOURCES[_cat] + [r"анықтама-кітапша", r"freshman"]

LANGS = ("ru", "kk", "en")


def main():
    rows = []
    for cat, items in Q.items():
        for i, triple in enumerate(items, start=1):
            for lang, q in zip(LANGS, triple):
                rows.append({"id": f"{cat}_{i:02d}_{lang}", "pair_id": f"{cat}_{i:02d}", "category": cat,
                             "lang": lang, "question": q, "expected_sources": SOURCES[cat]})
    with open(HERE / "questions.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(HERE / "out_of_scope.jsonl", "w", encoding="utf-8") as f:
        for i, triple in enumerate(OUT_OF_SCOPE, start=1):
            for lang, q in zip(LANGS, triple):
                f.write(json.dumps({"id": f"oos_{i:02d}_{lang}", "lang": lang, "question": q,
                                    "expected_status": "out_of_scope"}, ensure_ascii=False) + "\n")
    with open(HERE / "unanswerable.jsonl", "w", encoding="utf-8") as f:
        for i, triple in enumerate(UNANSWERABLE, start=1):
            for lang, q in zip(LANGS, triple):
                f.write(json.dumps({"id": f"una_{i:02d}_{lang}", "lang": lang, "question": q,
                                    "expected_status": "not_found"}, ensure_ascii=False) + "\n")
    print(f"questions: {len(rows)}, out-of-scope: {len(OUT_OF_SCOPE) * 3}, unanswerable: {len(UNANSWERABLE) * 3}")


if __name__ == "__main__":
    main()
