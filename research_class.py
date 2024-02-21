import phik
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from prophet import Prophet
from datetime import timedelta
from statsmodels.tsa.stattools import adfuller
from phik.report import plot_correlation_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from statsmodels.tsa.seasonal import seasonal_decompose




# установка констант
RANDOM_STATE = 42

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

	def add_new_features(self, dataset, holidays):
		'группировка по client_id и date с аггрегаций остальных признаков'
		grouped_dataset = (dataset
						   .groupby(['client_id', 'date'])
						   .agg({'quantity': 'sum', 'price': 'sum', 'message_id': 'sum'})
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
		grouped_dataset = grouped_dataset[
		(
			(grouped_dataset['date'] >= (grouped_dataset['date'].max() - timedelta(days=30))) & 
			(grouped_dataset['target'] == 1)
		) | 
		(grouped_dataset['date'] < grouped_dataset['date'].max() - timedelta(days=30))
		]

		'''
  установка даты в индекс
  '''
		grouped_dataset.set_index('date', inplace=True)
		grouped_dataset.sort_index(inplace=True)

		# Соотношение классов целевого признака
		sizes = [grouped_dataset['target'].value_counts()[1], grouped_dataset['target'].value_counts()[0]]
		fig1, ax1 = plt.subplots()
		ax1.pie(sizes, labels=['True', 'False'], autopct='%1.0f%%')
		plt.title('Соотношение классов целевого признака', size=12)
		plt.show

		# Проверка корреляций между признаками
		phik_overview = grouped_dataset.drop(columns=['client_id', 'message_id']).phik_matrix()
		sns.set()
		plot_correlation_matrix(phik_overview.values,
                        x_labels=phik_overview.columns,
                        y_labels=phik_overview.index,
                        vmin=0, vmax=1,
                        fontsize_factor=0.8, figsize=(8, 8))
		plt.xticks(rotation=45)
		plt.title('Корреляция между признаками', fontsize=12, y=1.02)
		plt.tight_layout()

		return grouped_dataset

	def seasonality_and_stationarity(self, dataset, period_1, period_2):
		'''разложение на тренд, сезонность и остатки за весь период'''
		
		decomposed_units_year = seasonal_decompose(dataset['target'], period=period_1)

		plt.figure(figsize=(10,6))
		plt.suptitle('Decomposition Analysis of Annual Data', fontsize=12)
		plt.subplot(311)
		decomposed_units_year.trend.plot(ax=plt.gca())
		plt.title('Trend')
		plt.subplot(312)
		decomposed_units_year.seasonal.plot(ax=plt.gca())
		plt.title('Seasonality')
		plt.subplot(313)
		decomposed_units_year.resid.plot(ax=plt.gca())
		plt.title('Residuals')
		plt.tight_layout()

		'''разложение на тренд, сезонность и остатки за две недели'''

		decomposed_units_month = seasonal_decompose(dataset['target']['2023-06-24':'2023-07-24'], period=period_2)

		plt.figure(figsize=(10,6))
		plt.suptitle('Decomposition Analysis of Monthly Data (June 24, 2023 - July 24, 2023)', fontsize=12)
		plt.subplot(311)
		decomposed_units_month.trend.plot(ax=plt.gca())
		plt.title('Trend')
		plt.subplot(312)
		decomposed_units_month.seasonal.plot(ax=plt.gca())
		plt.title('Seasonality')
		plt.subplot(313)
		decomposed_units_month.resid.plot(ax=plt.gca())
		plt.title('Residuals')
		plt.tight_layout()

		'''проверка стационарности временного ряда'''
		print('Проведение теста Дики-Фуллера для проверки ряда на стационарность')
		H0 = 'ряд стационарен, единичных корней нет'
		H1 = 'ряд не стационарен, имеются единичные корни'

		test = adfuller(dataset['target'])
		print('adf: ', test[0])
		print('p-value: ', test[1])
		print('Critical values: ', test[4])
		if test[0] > test[4]['5%']:
			print(H1)
		else:
			print(H0)

	def prepare_for_training(self, dataset):
		'''
  - на вход получает датасет
  - производит отделение целевого признака,
    масштабирование данных,
    разделение на две выборки
  - на выходе: две выборки, объект TimeSeriesSplit
    печать размерностей этих выборок
  '''
		
		y = dataset['target']
		X = dataset.drop(['target'], axis=1)

		# Приведение данных к единому масштабу
		scaler = StandardScaler()
		X_es = (pd.DataFrame(scaler.fit_transform(X.drop(['message_id', 'client_id'], axis=1)),
							 columns=X.drop(['message_id', 'client_id'], axis=1).columns,
							 index=X.drop(['message_id', 'client_id'], axis=1).index))
		X_es = pd.concat([X_es, X[['message_id', 'client_id']]], axis=1)
		# Разделение данных на выборки
		X_train, X_test, y_train, y_test = (train_test_split(X_es,
															 y,
															 test_size=0.1,
															 random_state=RANDOM_STATE,
															 shuffle=False))

		tscv = TimeSeriesSplit(n_splits=round((X_train.shape[0] / X_test.shape[0]) - 1))
		print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

		return X_train, X_test, y_train, y_test, tscv
    
	def train(self):
		# Метод для обучения модели на подготовленных данных
		# Здесь можно использовать любой алгоритм машинного обучения для обучения модели
		pass
    
	def feature_importance(self):
		# Метод для определения важности признаков
		# Здесь можно использовать различные методы, например, анализ важности признаков с помощью модели или перестановочный важность
		pass