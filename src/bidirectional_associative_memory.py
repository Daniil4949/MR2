# Лабораторная работа 2 по дисциплине МРЗвИС
# Выполнена студентом группы 121702 БГУИР Кимстач Д.Б.
# Реализация модели двунаправленной ассоциативной памяти
# Вариант 11
# Ссылки на источники:
# https://rep.bstu.by/bitstream/handle/data/30365/471-1999.pdf?sequence=1&isAllowed=y


import numpy as np


class BAM:
    def __init__(self, input_size, output_size, learning_rate=0.8):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.W = np.zeros((input_size, output_size))

        self.epochs = 0

    @staticmethod
    def _activation_function(x, previous_x):
        """
        Функция активации с учетом условий:
        1, если x > 0
        -1, если x < 0
        previous_x, если x == 0
        """
        result = np.where(x > 0, 1, np.where(x < 0, -1, previous_x))
        return result

    def train(self, input_vectors, output_vectors):
        for x, y in zip(input_vectors, output_vectors):
            self.W += self.learning_rate * np.outer(x, y)
            self.epochs += 1
            print(f"Эпоха {self.epochs}")

    def recall(self, input_data=None, output_data=None, max_iterations=100):
        if input_data is not None and output_data is None:
            output_data = self._activation_function(np.dot(input_data, self.W), np.zeros(self.output_size))
        elif output_data is not None and input_data is None:
            input_data = self._activation_function(np.dot(output_data, self.W.T), np.zeros(self.input_size))

        for iteration in range(max_iterations):
            new_input = self._activation_function(np.dot(output_data, self.W.T), input_data)
            new_output = self._activation_function(np.dot(new_input, self.W), output_data)

            if np.array_equal(new_input, input_data) and np.array_equal(new_output, output_data):
                break

            input_data, output_data = new_input, new_output

        return input_data, output_data


def image_beautiful_print(image, rows, cols):
    image = np.sign(image).astype(np.int_)
    image = image.astype(np.object_)
    image[image == 1] = '⬜'
    image[image == -1] = '⬛'
    image = image.reshape(rows, cols)
    image_list = image.tolist()
    for row in image_list:
        print(''.join(row))
