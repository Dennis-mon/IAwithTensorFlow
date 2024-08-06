const { IsolationForest } = require('isolation-forest-visualization');
const loadCSVPercentage = require('./loadCSVPercentage')

//libria para crear el canvas
const { createCanvas } = require('canvas');

//libreria para crear los graficos
const {Chart, registerables} = require('chart.js');

//libreria para exportar en png el grafico
const fs = require('fs');

//librerias para adaptar fechas para el canvas
require('chartjs-adapter-moment');
const moment = require('moment');

// Registrar todos los componentes necesarios
Chart.register(...registerables);

//ruta de los CSV
const pathCSV = '../csv/';

//Cargamos CSV
//en este caso usamos dataColumns para los datos de entrenamiento en un futuro
//labelColum la utilizamos para organizar el grafico, pero no para los entrenamientos
const { features , labels, testFeatures, testLabels } = loadCSVPercentage(
    `${pathCSV}ejemplo-isolation-forest.csv`,
    ',',
    {
        shuffle: false,
        percentageTest: 1,
        dataColumns: ['dia de la semana', '0-24','value'],
        labelColumns: ['año','mes','dia','hora','minuto']
    }
)

const data = [[7, 3], [2, 5], [3, 4]];

//creamos isolation forest
var myForest = new IsolationForest(data, 100,  data.length);

const anomalyScores = myForest.dataAnomalyScore(2);

console.log(anomalyScores);

// export the forest
myForest.exportForest('png', 'forestExport');






/*




//creamos array donde colocaremos objetos con el indice y valor de anomalia de los elementos raros.
var datos = [];

//filtramos los anomalyScore mayores a 0.5
scores.forEach( (element, index) =>{

    //si supera 0.5 de valor -> es anomalia = true
    //sino -> no es anomalia = false
    let tipo = (element > 0.5) ? true : false;

    let value = features[index][2];
    date = new Date( labels[index][0],labels[index][1],labels[index][2],labels[index][3],labels[index][4]*15);
    datos.push([{"valor": value, "date": date}, tipo]);
})

//ordenamos por fechas
datos.sort((a, b) => a[0].date - b[0].date);

//cambiamos las fechas a string para poder representarlas en el canvas
datos.map( element => { element[0].date = moment(element[0].date).format('YYYY-MM-DD HH:mm')  } );

//mostrar datos en terminal
console.log("scores", scores.length);
console.log("datos", datos.length);
console.log("datos", datos);

//Creamos un array para representarlo en el canvas con los datos que queremos en el eje (x,y)
var data = [];
datos.forEach( element => {
    data.push( {x: element[0].date , y: element[0].valor } )
});

// Configuración del lienzo (canvas)
const canvas = createCanvas(2000,2000);
const ctx = canvas.getContext('2d');

const chart = new Chart(ctx, {
    type: 'line',
    data: {
        datasets: [
            {
                label: 'Datos normales',
                data: data,
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1,
                pointRadius: (context) => (datos[context.dataIndex][1]) ? 15 : 5 ,                 //anomalia = 10, NO anomalia = 5
                pointBackgroundColor: (context) => (datos[context.dataIndex][1]) ? 'red' : 'blue', //anomalia = red, NO anomalia = blue
                showLine: true
            },
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            x: {
                type: 'time',
                time: {
                    unit: 'minute', // Otras unidades: 'month', 'year', etc.
                    displayFormats: {
                        minute: 'MMM DD HH:mm'
                    },
                    tooltipFormat: 'YYYY-MM-DD HH:mm',
                },
                title:{
                    display: true,
                    text: 'fecha'
                },
                ticks:{
                    stepSize:16
                }
            },
            y: {
                title: {
                    display: true,
                    text: 'valor'
                }
            }
        }
    }
});

// Guarda el gráfico como una imagen PNG
const buffer = canvas.toBuffer('image/png');
fs.writeFileSync('chart.png', buffer);
console.log('Gráfico guardado en chart.png');






*/