from random import randint, choice, random

# объявляем переменные
TARGET = 'java'
GENERATION_NUM = 100
POPULATION_NUM = 100
MUTATION_RATE = 0.01
matting_pool = []
population = []

# создаём первую популяцию
for _ in range(POPULATION_NUM):
    population.append({
        'length': len(TARGET),
        'genes': [randint(97, 122) for __ in range(len(TARGET))],
        'fitness': 0
    })

# оцениваем пригодность
for p in population:
    for g in p['genes']:
        if chr(g) in TARGET:
            p['fitness'] += 1
    p['fitness'] /= 100

for generation in range(GENERATION_NUM):
    # подготавливаем набор для скрещиванию
    for p in population:
        for _ in range(int(p['fitness'] * 100)):
            matting_pool.append(p)

    # скрещивание
    for idx in range(len(population)):
        # выбираем случайных родителей
        parent_a = choice(matting_pool)
        parent_b = None

        while not parent_b and parent_a != parent_b:
            parent_b = choice(matting_pool)

        # crossover
        midpoint = randint(0, len(parent_a['genes']))
        child = {
            'length': len(TARGET),
            'genes': [randint(97, 122) for __ in range(len(TARGET))],
            'fitness': 0
        }

        for i in range(len(parent_a['genes'])):
            if i > midpoint:
                child['genes'][i] = parent_a['genes'][i]
            else:
                child['genes'][i] = parent_b['genes'][i]

        # mutations
        for i in range(len(child['genes'])):
            if MUTATION_RATE > random():
                child['genes'][i] = randint(97, 122)

        # оцениваем пригодность
        for g in child['genes']:
            if chr(g) in TARGET:
                child['fitness'] += 1
        child['fitness'] /= 100

        population[idx] = child

for p in population:
    string = ''

    for g in p['genes']:
        string += chr(g)

    print(string)
