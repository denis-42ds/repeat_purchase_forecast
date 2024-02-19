import pandas as pd
import seaborn as sns
from datetime import timedelta
import matplotlib.pyplot as plt

class DatasetExplorer:
	def __init__(self, dataset, y_lim):
		self.dataset = dataset
		self.y_lim = y_lim

	def explore_dataset(self):
		# Вывод информации о датасете
		self.dataset.info()

		# Вывод случайных примеров из датасета
		display(self.dataset.sample(5))

		# Количество полных дубликатов строк
		print(f"количество полных дубликатов строк: {self.dataset.duplicated().sum()}")

		# Круговая диаграмма для количества полных дубликатов
		sizes = [self.dataset.duplicated().sum(), self.dataset.shape[0]]
		fig1, ax1 = plt.subplots()
		ax1.pie(sizes, labels=['duplicate', 'not a duplicate'], autopct='%1.0f%%')
		plt.title('Количество полных дубликатов в общем количестве строк', size=12)
		plt.show()

		# Удаление полных дубликатов строк
		#  self.dataset.drop_duplicates(inplace=True)
		#  self.dataset.reset_index(drop=True, inplace=True)

		# Количество уникальных значений client_id
		print(f"""количество уникальных значений client_id:
		{self.dataset['client_id'].nunique()}""")
		print(f"""количество уникальных значений client_id в общем количестве client_id:
		{self.dataset['client_id'].nunique() / self.dataset.shape[0] * 100:.3f}%""")

		# Круговая диаграмма для соотношения уникальных и повторяющихся значений client_id
		sizes = [self.dataset['client_id'].nunique(), self.dataset.shape[0]]
		fig1, ax1 = plt.subplots()
		ax1.pie(sizes, labels=['unique values', 'duplicate values'], autopct='%1.0f%%')
		plt.title('Соотношение уникальных и повторяющихся значений client_id', size=12)
		plt.show()

		# Количество уникальных значений message_id
		print(f"""количество уникальных значений message_id:
		{self.dataset['message_id'].nunique()}""")
		print(f"""количество уникальных значений message_id в общем количестве message_id:
		{self.dataset['message_id'].nunique() / self.dataset.shape[0] * 100:.3f}%""")

		# Круговая диаграмма для соотношения уникальных и повторяющихся значений message_id
		sizes = [self.dataset['message_id'].nunique(), self.dataset.shape[0]]
		fig1, ax1 = plt.subplots()
		ax1.pie(sizes, labels=['unique values', 'duplicate values'], autopct='%1.0f%%')
		plt.title('Соотношение уникальных и повторяющихся значений message_id', size=12)
		plt.show()

		# Описание столбца quantity
		print(f"""описание quantity:
		{self.dataset['quantity'].describe()}""")
		print(f"""количество уникальных значений quantity: {self.dataset['quantity'].nunique()}""")

		# График распределения количества товаров
		sns.kdeplot(data=self.dataset, x='quantity', clip=(-1, 5), fill=True)
		plt.title('Распределение количества товаров', size=12)
		plt.show()

		# Описание столбца price
		print(f"""описание price:
		{self.dataset['price'].describe()}""")
		print(f"""количество уникальных значений price: {self.dataset['price'].nunique()}""")

		# График распределения цены товаров
		sns.kdeplot(data=self.dataset, x='price', clip=(-1, self.y_lim), fill=True)
		plt.title('Распределение цены товаров', size=12)
		plt.show()

		# Диаграмма размаха для цены товаров
		plt.figure(figsize=(8, 6))
		sns.boxplot(data=self.dataset['price'])
		plt.ylim(-1, self.y_lim)
		plt.title('Диаграмма размаха для цены товаров', size=12)
		plt.ylabel('Количество, шт')
		plt.show()

		# Интервал записей в датафрейме
		print(f"""Первая запись в датафрейме: {self.dataset['date'].min()}
		Последняя запись в датафрейме: {self.dataset['date'].max()}""")

		# Вывод информации о датасете после исследований
		#  print("Информация о датасете после первичных преобразований:")
		#  self.dataset.info()

	def add_new_features(self, holidays):
		'группировка по client_id и date с аггрегаций остальных признаков'
		grouped_dataset = (self.dataset
                           .groupby(['client_id', 'date'])
                           .agg({'quantity': 'sum', 
                                 'price': 'sum', 
                                 'message_id': 'sum'})
                           .reset_index()
                          )
		'''
  создание целевого признака:
  - если в течение 30-ти дней совершена покупка - 1, нет - 0
  '''
		grouped_dataset['target'] = 0
		prev_client_id = None
		prev_purchase_date = None

		for index, row in grouped_dataset.iterrows():
			if prev_client_id == row['client_id'] and (row['date'] - prev_purchase_date).days <= 30:
				grouped_dataset.at[index-1, 'target'] = 1
			else:
				grouped_dataset.at[index, 'target'] = 0

			prev_client_id = row['client_id']
			prev_purchase_date = row['date']

		'''
  создание признака с накопительной суммой активностей в течение 30-ти дней
  '''
		grouped_dataset['cumulative_sum'] = 0
		prev_client_id = None
		prev_purchase_date = None
		cumulative_sum = 0

		for index, row in grouped_dataset.iterrows():
			if prev_client_id == row['client_id'] and (row['date'] - prev_purchase_date).days <= 30:
				cumulative_sum += 1
			else:
				cumulative_sum = 1

			grouped_dataset.at[index, 'cumulative_sum'] = cumulative_sum

			prev_client_id = row['client_id']
			prev_purchase_date = row['date']
		'''
  добавление данных о выходных и праздничных днях
  '''
		grouped_dataset['is_holiday'] = grouped_dataset['date'].isin(holidays['date']).astype(int)
		'''
  исключение из датасета последних 30-ти дней, т.к. в этот период нельзя посчитать таргет
  '''
		grouped_dataset = (
			grouped_dataset[(grouped_dataset['date'] < (grouped_dataset['date'].max() - timedelta(days=30))) & 
			(grouped_dataset['target'] == 0)]
		)
		'''
  установка даты в индекс
  '''
		grouped_dataset.set_index('date', inplace=True)
		grouped_dataset.sort_index(inplace=True)

		return grouped_dataset

	def prepare_for_training(self):
		# Метод для подготовки датасета к обучению
		# Здесь можно выполнить предобработку данных, например, заполнение пропущенных значений, кодирование категориальных признаков и т.д.
		pass
    
	def train(self):
		# Метод для обучения модели на подготовленных данных
		# Здесь можно использовать любой алгоритм машинного обучения для обучения модели
		pass
    
	def feature_importance(self):
		# Метод для определения важности признаков
		# Здесь можно использовать различные методы, например, анализ важности признаков с помощью модели или перестановочный важность
		pass