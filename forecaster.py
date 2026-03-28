import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

np.random.seed(42)
data = []
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
hours = range(10, 20)

for day in range(60):
    day_of_week = days[day % 5]
    for hour in hours:
        wait_time = random.randint(5, 15)
        
        if hour in [13, 14]:
            wait_time += random.randint(15, 30)
        if hour == 17:
            wait_time += random.randint(10, 20)
        if day_of_week == 'Friday':
            wait_time += random.randint(5, 10)
            
        data.append([day_of_week, hour, wait_time])

df = pd.DataFrame(data, columns=['Day_of_Week', 'Hour_of_Day', 'Wait_Time_Minutes'])
df.to_csv('underbelly_data.csv', index=False)
print("Dataset 'underbelly_data.csv' created successfully!")

le = LabelEncoder()
df['Day_of_Week_Encoded'] = le.fit_transform(df['Day_of_Week'])

X = df[['Day_of_Week_Encoded', 'Hour_of_Day']]
y = df['Wait_Time_Minutes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeRegressor(max_depth=5, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"Model Mean Absolute Error: {mae:.2f} minutes")

test_day = le.transform(['Wednesday'])[0]
test_hour = 13
predicted_wait = model.predict([[test_day, test_hour]])

print(f"Predicted wait time at the Underbelly for Wednesday at 1 PM: {predicted_wait[0]:.0f} minutes")
