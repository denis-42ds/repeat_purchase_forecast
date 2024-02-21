# импорт модулей
import phik
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from prophet import Prophet
from datetime import timedelta
from catboost import CatBoostClassifier
from statsmodels.tsa.stattools import adfuller
from phik.report import plot_correlation_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import make_scorer, roc_curve, recall_score, roc_auc_score, precision_score, confusion_matrix



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

	def add_new_features(self, dataset, holidays):
		'кодирование message_id при помощи OrdinalEncoder'
		encoder = OrdinalEncoder()
		dataset['message_id_encoded'] = encoder.fit_transform(dataset[['message_id']])
		dataset.drop(columns=['message_id'], inplace=True)
		'группировка по client_id и date с аггрегаций остальных признаков'
		grouped_dataset = (dataset
						   .groupby(['client_id', 'date'])
						   .agg({'quantity': 'sum', 'price': 'sum', 'message_id_encoded': 'sum'})
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
		phik_overview = grouped_dataset.drop(columns=['client_id']).phik_matrix()
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
		X_es = (pd.DataFrame(scaler.fit_transform(X.drop(['client_id'], axis=1)),
							 columns=X.drop(['client_id'], axis=1).columns,
							 index=X.drop(['client_id'], axis=1).index))
		X_es = pd.concat([X_es, X[['client_id']]], axis=1)
		# Разделение данных на выборки
		X_train, X_test, y_train, y_test = (train_test_split(X_es,
															 y,
															 test_size=0.1,
															 random_state=RANDOM_STATE,
															 shuffle=False))

		tscv = TimeSeriesSplit(n_splits=round((X_train.shape[0] / X_test.shape[0]) - 1))
		print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

		return X_train, X_test, y_train, y_test, tscv
    
	def modeling_pipeline(self, model_name, X_train, y_train, tscv, periods):
		'''
  - на вход получает название модели, обучающие данные и объект TimeSerisSplit;
  - на выходе выводит на печать recall и precision, параметры модели и лучшую метрику;
  - визуализирует результаты обучения на диаграмме.
  '''
		if model_name == 'Baseline':
			model = LogisticRegression()
			scorers = {
				'recall': make_scorer(recall_score),
				'precision': make_scorer(precision_score)
			}
			param_grid = {'fit_intercept': [True, False]}
			grid_search = GridSearchCV(model, param_grid, cv=tscv, scoring=scorers, refit='recall')
			grid_search.fit(X_train.drop(['client_id'], axis=1).values, y_train)
			y_pred = grid_search.predict(X_train.drop(['client_id'], axis=1).values)
			y_pred_proba = grid_search.predict_proba(X_train.drop(['client_id'], axis=1).values)[:, 1]
			print("Лучшие параметры GridSearch:", grid_search.best_params_)
			print("Лучшая оценка GridSearch:", abs(grid_search.best_score_))
		
		elif model_name == 'Prophet':
			model = Prophet()
			df = pd.DataFrame({'ds': X_train.index, 'y': y_train.values})
			model.fit(df)
			future = model.make_future_dataframe(periods=periods)
			forecast = model.predict(future)
			threshold = 0.5
			y_pred = np.where(forecast['yhat'].iloc[-len(y_train):].values > threshold, 1, 0)
			# применение калибровки вероятностей к прогнозам, чтобы получить вероятности принадлежности к классу 1
			calibrated_model = CalibratedClassifierCV()
			calibrated_model.fit(y_pred.reshape(-1, 1), y_train)
			y_pred_proba = calibrated_model.predict_proba(y_pred.reshape(-1, 1))[:, 1]
		
		elif model_name == 'CatBoost':
			model = CatBoostClassifier(random_state=RANDOM_STATE, eval_metric='AUC')
			model.fit(X_train.drop(['client_id'], axis=1).values, y_train, verbose=100)
			y_pred_proba = model.predict_proba(X_train.drop(['client_id'], axis=1).values)[:, 1]
			y_pred = model.predict(X_train.drop(['client_id'], axis=1).values)

		# Вычисление ROC-AUC, precision, recall
		roc_auc_value = roc_auc_score(y_train, y_pred_proba)
		recall = recall_score(y_train, y_pred, pos_label=1, zero_division='warn')
		precision = precision_score(y_train, y_pred, pos_label=1, zero_division='warn')

		print("ROC-AUC:", roc_auc_value)
		print("Precision:", precision)
		print("Recall:", recall)

		# Визуализация кривой ROC
		fpr, tpr, thresholds = roc_curve(y_train, y_pred_proba)
		sns.set_style('darkgrid')
		plt.plot(fpr, tpr, linewidth=1.5, label='ROC-AUC (area = %0.2f)' % roc_auc_value)
		plt.plot([0, 1], [0, 1], linestyle='--', linewidth=1.5, label='random_classifier')
		plt.xlim([-0.05, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate', fontsize=11)
		plt.ylabel('True Positive Rate', fontsize=11)
		plt.title('%s Receiver Operating Characteristic' % model_name, fontsize=12)
		plt.legend(loc='lower right')
		plt.show()

		return recall, precision, roc_auc_value, model
    
	def test_best_model(self, model, X_train, X_test, y_test):
		'''
  принимает на вход модель, обучающую и тестовую выборки, тестовый таргет
  возвращает:
  - матрицу ошибок
  - метрики для тестовой выборки
  - диаграмму важности признаков
  '''
		try:
			if len(X_test) < 1 or len(np.unique(y_test)) != 2:
				raise ValueError("Невозможно посчитать метрики - в данных присутствует только один класс")
				
			y_proba = model.predict_proba(X_test)[:, 1]
			roc_auc = roc_auc_score(y_test, y_proba)
			y_pred = model.predict(X_test)
			recall = recall_score(y_test, y_pred, pos_label=1)
			precision = precision_score(y_test, y_pred, pos_label=1)
        
			print(f"ROC-AUC на тестовой выборке: {round(roc_auc, 2)}")
			print(f"Precision на тестовой выборке: {round(precision, 2)}")
			print(f"Recall на тестовой выборке: {round(recall, 2)}")

			sns.set_style('darkgrid')
			plt.figure(figsize=(6, 6))
			sns.heatmap(confusion_matrix(y_test, y_pred.round()), annot=True, fmt='3.0f', cmap='crest')
			plt.title('Матрица ошибок', fontsize=12, y=1.02)
			plt.show()

			features_importance = (
				pd.DataFrame(data = {'feature': X_train.drop('client_id', axis=1).columns, 
									 'percent': np.round(model.feature_importances_, decimals=1)})
			)
			plt.figure(figsize=(8, 6))
			plt.bar(features_importance.sort_values('percent', ascending=False)['feature'], 
					features_importance.sort_values('percent', ascending=False)['percent'])

			plt.xticks(features_importance['feature'])
			plt.xticks(rotation=45)
			plt.ylabel('Процент от общего значения')
			plt.title("Важность признаков", fontsize=12, y=1.02)
			plt.show()
            
		except ValueError as e:
			print(e)