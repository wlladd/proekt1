 надо создать плагины /индикаторы по следующей спецификации параболик сар , задается текущий таймфрейм, задается старший таймрейм , данные выводятся в формате его положения выше цены 1 ниже цены -1 , получается он будет выдавать 2 колонки положение на исходном таймфрейме ли +1 или -1 и вторая колонка положение на стршем таймфрейме или +1 или -1
Индикатор адх, в нем нас интересует лини +ди и -ди их значения будут такого вида +1 если квеличилось относительно прошлого значения и -1 если умешилось эти значения для каждой линии , такк же будут собираться положение линий относительно друг друга столбец будут называться как нибудт DI+/DI- и если ди + выше тобудет значение +1 если ниже то -1 следущий столбец будет для каждой лини записывать ее первую смену направления, если первый раз после движения вниз или вверх линия закроется в противоположную сторону то тогда 1 (это будет обозначать первую ее смену направления) а потом будут идти 0 до следующей первой смены , 

Следующий будет наборы скользящих средних , в них мы тоже будет записывать +1 если выше цены находится -1 если ниже 

Стандартный индикатор rsi в нем мы дудем записывать закрылся выше предыддущего или ниже , +1+,-1, пересечение уровней 20 и 80  если выши/80 то +1 на всех значениях которые выше если ниже то -1 на всех значениях которые ниже  то же самое и про уровень 20, и отдельно мы будем считать дивергенцию за период 20 свечей , если дивергенция на покупку то +1 если на продажу то -1 и того у нас получаться --1 стобец рсе это он больше или меньше предыдущего(+1 или-1)  уровень 20 (+1или-1) уровень 80(+1или-1) дивергенция (+1или-1)  всего 4 столбца 
Индикатор свечных патернов с возможностью выбора этих патернов , все или несколько или вообще какой нибудь один , каждый столбец будет называться названием патерна, если его нет то будет записываться 0 если этот патерн на покупку то +1 , если этот патерн на продажу то -1 

Индикатор зиг заг с натраиваемым значением и его колонки будут - одна будут записывать его закрытый минимум , до закрытия следующего, потом  наченет записывать следующий и вторая колонка так же будет записывать его закрытый максимум до появления следующего 

your_project/
├── app.py
└── plugins/
    └── indicators/
        ├── ParabolicSAR/
        │   ├── config.json
        │   └── plugin.py
        ├── ADX_DI/
        │   ├── config.json
        │   └── plugin.py
        ├── MovingAverages/
        │   ├── config.json
        │   └── plugin.py
        ├── RSI_Custom/
        │   ├── config.json
        │   └── plugin.py
        ├── CandlestickPatterns/
        │   ├── config.json
        │   └── plugin.py
        └── ZigZag/
            ├── config.json
            └── plugin.py

теперь нам надо создать пайтон с графичесским интерфейсом который подключит наши плгины индикаторы, подключит наши сохраненные данные формата csv или json , так же подтянет данные стакана цен (все это из наших сохраненных файлов пример которых я показывал)  и на выходе сохранит csv или json с расчитанными индикаторами, правильно встроенным стаканом цен , отчищенными пустыми данными , для того чтобы сразу можно было использовать для обучения нейросети , графичесский интерфейс лучше вэб, он наиболее удобен несколько замечаний, зачем параболик сар требует старший таймфрейм если он может расчитать его на основании младшего? + в его  выборе надо дать возможность пользователю именно выбрать какой именно старший таймфрейм , и по свечным моделям! там в синтаксисе во первых ведь по другому , им вр втрых там сразу можно вывод всех моделей сделать , и можно через меню пользователю выбрать какие именно модели он добавит , ведь так? 


в таком виде поступают данные по свечному стакану - US100_depth_1746057600000_1748736000000.json
а в таком виде поступают данные по свечам , для расчета индикаторов/плагинов - US100_klines_1746057600000_1748736000000.json
в таком виде мы получаем выходные данные , есть 0 ошибочные строки для стакана(на самом деле там должны быть данные)  features_dataset (3).json