import streamlit as st
import pandas as pd
from surprise import SVD, Reader, Dataset
from surprise.model_selection import cross_validate, GridSearchCV

# Se crea DataFrame vacío
df_usuario = pd.DataFrame(columns=["id", "recipes", "n_recipes", "ratings", "n_ratings"])

# ... (el resto del código)

# Se agregan las filas especificadas
data = [
    {"id": 34, "recipes": ["http://www.edamam.com/ontologies/edamam.owl#recipe_2e9b699be433fab7da069629e1699455"], "n_recipes": 1, "ratings": [5.0], "n_ratings": 1},
    {"id": 58, "recipes": ["http://www.edamam.com/ontologies/edamam.owl#recipe_067f0b7be628ae847366e4f3e614b319", "http://www.edamam.com/ontologies/edamam.owl#recipe_3bc095c814af01cfc5e12aa3c3bad9e6", "http://www.edamam.com/ontologies/edamam.owl#recipe_88c93d34a2f0c4a9ebf55c8d7c985458"], "n_recipes": 3, "ratings": [5.0, 4.0, 5.0], "n_ratings": 3},
    {"id": 10, "recipes": ["http://www.edamam.com/ontologies/edamam.owl#recipe_ab7b274349df0ce399cd05b5167d7052", "http://www.edamam.com/ontologies/edamam.owl#recipe_3d45f44e2e398b038c1113b0fd9a484c"], "n_recipes": 2, "ratings": [5.0, 3.0], "n_ratings": 2}
]

df_usuario = pd.DataFrame(data)
# Df resultante
df_usuario

def getRecipeRatings(idx):
    user_recipes = [item for item in df_usuario.loc[idx]['recipes']]
    user_ratings = [float(rating) for rating in df_usuario.loc[idx]['ratings']]
    df = pd.DataFrame(list(zip(user_recipes, user_ratings)), columns=['Recipe', 'Rating'])
    df.insert(loc=0, column='User', value=df_usuario.loc[idx].name)  # Use the index as the 'User' value
    return df

recipe_ratings = pd.DataFrame(columns=['User', 'Recipe', 'Rating'])

for idx, row in df_usuario.iterrows():
    recipe_ratings = pd.concat([recipe_ratings, getRecipeRatings(idx)], ignore_index=True)
recipe_ratings

# Se indican cuales son los valores del rating
reader = Reader(rating_scale=(0, 5))

# Se indican las columnas del df que usará
data_1 = Dataset.load_from_df(recipe_ratings[["User", "Recipe", "Rating"]], reader)

from surprise.model_selection import GridSearchCV

param_grid = {'n_factors': [100, 150],
              'n_epochs': [20, 25, 30],
              'lr_all': [0.005, 0.01, 0.1],
              'reg_all': [0.02, 0.05, 0.1]}
grid_search = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
grid_search.fit(data_1)
print(grid_search.best_score['rmse'])
print(grid_search.best_score['mae'])
print(grid_search.best_params['rmse'])
svd_param = grid_search.best_estimator['rmse']

cross_validate(svd_param, data_1, measures=['RMSE', 'MAE'], cv=5, verbose=True)
cv_results_svd_param = cross_validate(svd_param, data_1, measures=['RMSE', 'MAE'], cv=5, verbose=True)
mae_svd_param = cv_results_svd_param['test_mae'].mean()
rmse_svd_param = cv_results_svd_param['test_rmse'].mean()

user_id = 0  # Reemplaza 0 con el ID del usuario para el cual deseas hacer recomendaciones

# Obtener las recetas que el usuario aún no ha calificado
unrated_recipes = [recipe for recipe in recipe_ratings['Recipe'].unique() if recipe not in df_usuario.loc[user_id]['recipes']]

# Realizar predicciones para las recetas no calificadas
predictions = [svd_param.predict(user_id, recipe) for recipe in unrated_recipes]

# Ordenar las predicciones por la calificación estimada
sorted_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)

# Obtener las mejores N recomendaciones
N = 5  # Reemplaza con el número deseado de recomendaciones
top_n_recommendations = sorted_predictions[:N]

print(f"Top {N} Recomendaciones para el Usuario {user_id}:")
for i, prediction in enumerate(top_n_recommendations):
    print(f"{i + 1}. Receta: {prediction.iid}")



# Código de Streamlit
st.title("Recomendaciones de Recetas")

user_id = st.number_input("Ingrese el ID del usuario", min_value=0, max_value=len(df_usuario)-1, value=0, step=1)

if st.button("Generar Recomendaciones"):
    # Obtener las recetas que el usuario aún no ha calificado
    unrated_recipes = [recipe for recipe in recipe_ratings['Recipe'].unique() if recipe not in df_usuario.loc[user_id]['recipes']]

    # Realizar predicciones para las recetas no calificadas
    predictions = [svd_param.predict(user_id, recipe) for recipe in unrated_recipes]

    # Ordenar las predicciones por la calificación estimada
    sorted_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)

    # Obtener las mejores N recomendaciones
    N = 5  # Reemplaza con el número deseado de recomendaciones
    top_n_recommendations = sorted_predictions[:N]

    st.write(f"Top {N} Recomendaciones para el Usuario {user_id}:")
    for i, prediction in enumerate(top_n_recommendations):
        st.write(f"{i + 1}. Receta: {prediction.iid}")