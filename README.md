# Proyecto-PCD

Nuestro proyecto final es crear una página web empleando HTML, CSS, JavaScript & Python. En dicha página web habrá una gran variedad de recetas que vayan desde platillos sencillos, ya sea individuales, para citas o familiares; hasta recetas de repostería y coctelería. La página web será alimentada por medio de una API que contenga todas las recetas. La página web será de libre acceso, por lo que toda persona tendrá total libertad de ver las recetas como desee, así como también tendrá un sistema de registro / inicio de sesión que brinde a los usuarios registrados nuevas opciones como puntuar recetas, dejar comentarios en otras recetas y editar sus perfiles.

Para la creación de la página usaremos un modelo de machine learning que haga el funcionamiento de la página web como si fuera una especie de red social. El modelo usará un sistema de recomendaciones que muestre a los usuarios diferentes platillos en base a sus gustos. El modelo se basará en las puntuaciones, búsquedas, e ingredientes más usados por los usuarios, es decir, un modelo content-based. 

El repositorio cuenta con las siguientes carpetas:

- API: Aquí se encuentran los archivos necesarios para el desplegue del modelo usando streamlit
- Data: En esta carpeta se encuentran los datasets utilizados para el entrenamiento del modelo
- EDA: Aquí se hace un análisis exploratorio de los datos
- Training: Se encuentran los modelos hechos para el sistema de recomendación y como lo subimos a mlflow
- Informe escrito: En esta se encuentra un archivo ipynb donde se explica el proyecto más a detalle además de anclar las capturas de las pruebas del sistema funcionando
