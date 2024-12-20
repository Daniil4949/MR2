import numpy as np

from src.bidirectional_associative_memory import BAM, image_beautiful_print
from src.images import input_patterns, associations, output_patterns

if __name__ == '__main__':
    alphabet_inputs = np.array([input_patterns[key] for key in associations.keys()])
    alphabet_outputs = np.array([output_patterns[value] for value in associations.values()])

    input_size = len(alphabet_inputs[0])
    output_size = len(alphabet_outputs[0])

    bam = BAM(input_size, output_size, learning_rate=0.8)

    bam.train(alphabet_inputs, alphabet_outputs)

    # Демонстрация ассоциаций
    for input_symbol, output_symbol in associations.items():
        print(f"Входной образ ({input_symbol}):")
        image_beautiful_print(input_patterns[input_symbol], 4, 4)

        # Используем recall для ассоциации выходных данных
        predicted_input, predicted_output = bam.recall(input_data=np.array(input_patterns[input_symbol]))
        print(f"\nАссоциированный выходной образ ({output_symbol}):")
        image_beautiful_print(predicted_output, 4, 4)

        print(f"\nВосстановленный входной образ из выходного для {input_symbol}:")
        image_beautiful_print(predicted_input, 4, 4)
        print()

    # Тест с поврежденными данными
    print("Поврежденная буква А")
    damaged_A = [1, -1, 1, -1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1]
    image_beautiful_print(damaged_A, 4, 4)
    print()

    predicted_input, predicted_output = bam.recall(input_data=np.array(damaged_A))
    print("Ассоциация для поврежденной буквы А")
    image_beautiful_print(predicted_output, 4, 4)
