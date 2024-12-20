# Лабораторная работа 2 по дисциплине МРЗвИС
# Выполнена студентом группы 121702 БГУИР Кимстач Д.Б.
# Реализация модели двунаправленной ассоциативной памяти
# Вариант 11
# Ссылки на источники:
# https://rep.bstu.by/bitstream/handle/data/30365/471-1999.pdf?sequence=1&isAllowed=y


# Ассоциативные пары
associations = {
    'A': 'O',  # A ассоциируется с O
    'D': 'C',  # D ассоциируется с C
    'H': 'I',  # H ассоциируется с I
}

input_patterns = {
    'A': [-1, 1, 1, -1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1],
    'D': [1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1],
    'H': [1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, 1],

}

output_patterns = {
    'O': [-1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1],
    'C': [-1, 1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, 1],
    'I': [-1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1],
    'Z': [1, 1, 1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, 1, 1, 1],
}