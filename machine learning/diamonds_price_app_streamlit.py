import streamlit as st
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor ,GradientBoostingRegressor
from sklearn.metrics import mean_squared_error,r2_score
diamonds_df=sns.load_dataset("diamonds")

diamonds_df['cut'] = diamonds_df['cut'].astype("category").cat.codes
diamonds_df['color'] = diamonds_df['color'].astype("category").cat.codes
diamonds_df['clarity'] = diamonds_df['clarity'].astype("category").cat.codes


# Train the model
X = diamonds_df[['carat', 'cut', 'color', 'clarity']]
y = diamonds_df['price']
#train the multiple models
models={"random forest":RandomForestRegressor(n_estimators=100),"Gradient Boosting":
        GradientBoostingRegressor(n_estimators=100)}
for name,model in models.items():
    model.fit(X,y)

#sreate the steramlit app
st.title("diamond prediction")
st.sidebar.header("select a model")    

model_name=st.sidebar.select_slider("Model",options=list(models.keys()))
model=models[model_name]
y_pred=model.predict(X)
# calculate the model performance
mse=mean_squared_error(y,y_pred)
r2=r2_score(y,y_pred)
st.subheader(model_name)
st.write(f"MSE:{mse:.2f}")
st.write(f"R2:{r2:.2f}")