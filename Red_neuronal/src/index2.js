require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSVPercentage = require('./loadCSVPercentage.js');
const RedNeuronal2 = require('./red_neuronal2.js'); // Cargamos la clase encargada de realizar la regresión lineal
const { forEach } = require('lodash');
const pathCSV = '../csv/';

//Cargamos CSV 
const { features , labels, testFeatures, testLabels } = loadCSVPercentage(
    `${pathCSV}energia-enero-7dias.csv`,
    ',',
    {
        shuffle: true,
        percentageTest: 20,
        dataColumns: ['dia de la semana', '0-24'],
        labelColumns: ['value']
    }
)

/* CREAR REDES CON MUCHAS ACTIVACIONES

function Model( model, activation, error , resultado){
    this.model = model;
    this.activation = activation;
    this.error = error;
    this.resultado = resultado;
}

let models = [];

//=======================================
//Creamos una red para cada tipo de activacion
const activations = ['linear','sigmoid', 'tanh', 'relu'];

async function crearRedes2(){

    //creo red para cada activacion
    for (const typeActivation of activations) {

        const redneuronal = new RedNeuronal(
            features,
            labels,  
            {
                learningRate: 0.1,
                epochs: 2000,
                neurons: 10,
                activation: typeActivation
            }
        );

        redneuronal.compilar();
        const history = await redneuronal.entrenar();
        const porcentaje = redneuronal.testeo(testFeatures, testLabels);

        models.push( new Model( redneuronal, typeActivation, history.history.loss[ history.history.loss.length - 1 ] , porcentaje) ) 
    }

    // Ordenar el array por el valor de porcentaje en orden descendente
    models.sort(function(a, b) {
        return b.resultado - a.resultado;
    });

    
    models.forEach(model => {
        console.log("============");
        console.log("activacion:", model.activation);
        console.log("error:", model.error);
        console.log("resultado:", model.resultado);
    });

    console.log("==========Predicciones==========")

    models.forEach(element => {
        const resultado = element.model.prediccion(features);
        console.log("Resultado real:", labels);
        console.log("resultado prediccion:");
        resultado.print();

    });


}

crearRedes2()
*/

//PARA CREAR REDES NEURONALES UNA A UNA

//Creamos una nueva instancia de la clase Res Neuronal
const redneuronalPrueba = new RedNeuronal2(
    features, //Variables Independientes
    labels,   //Variables Dependientes
    //Configuración
    //learningRate = tasa de aprendizaje // por defecto 0.1
    //epochs       = nº de veces que se entrena el algoritmo para todos los datos // por fecefto 2000 
    {
        learningRate: 0.001,
        epochs: 50,
    },
    //InputLayer
        //neurons       -> nº de neuronas que queramos que tenga la capa, por defecto 1
        //activation    -> función de activación que vamos a usar en la capa, por defecto relu
    {
        neurons: 1,
        activation: 'relu'
    },
    //hidden layer      -> por defecto no hay ninguna capa de neuronas intermedias
        //neurons       -> nº de neuronas que queramos que tenga la capa
        //activation    -> función de activación que vamos a usar en la capa
    [
        {
            neurons: 32,
            activation: 'relu'
        },
        {
            neurons: 16,
            activation: 'relu'
        }

    ],
    //output layer
        //neurons       -> nº de neuronas que queramos que tenga la capa, por defecto 1
        //activation    -> función de activación que vamos a usar en la capa, por defecto relu
    {
        neurons: 1,
        activation: 'linear'
    }
);

//Función asincrona que se encarga de compilar y entrenar el algorimto ademas de generar una prediccón para unos valores independientes de entrada(features)
async function crearRedes(){

    redneuronalPrueba.compilar();
    await redneuronalPrueba.entrenar(); //await para impedir que el código se siga ejecutando hasta que el algoritmo este 100% entrenado

    //Guardamos los resultados de la predicción en una variable auxilar para poder verlos en la terminal
    const resultado = redneuronalPrueba.prediccion(features);
    console.log("Resultados obtenidos:");
    resultado.print();
    console.log("Resultados reales:");
    console.log(labels);
}

crearRedes();

//=======================================
