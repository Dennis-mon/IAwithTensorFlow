const IsolationForest = require('ml-isolation-forest');
const xl = require('excel4node');

let X = [
  [200, 50],
  [0.3, 0.1],
  [0.5, 0.3],
  [0.2, 0.1],
  [0.1, 0.1],
  [0.2, 0.05],
  [0.3, 0.3],
  [0.4, 0.2],
  [0.3, 0.4],
  [0.1, 0.1],
  [0.05, 0.1],
];

let anomalyDetector = new IsolationForest.IsolationForest();

anomalyDetector.train(X);

let result = anomalyDetector.predict([
  [200, 300],
  [0, 0.1],
  [0.2, 0.1],
  [0.1, 0.2],
]);

console.log(result);

//Creamos el Excel
var wb = new xl.Workbook();

//Ponemos la fecha
let date = new Date();
let fechaDia    = date.getUTCDate();
let fechaMes    = (date.getUTCMonth()) + 1; 
let fechaAño    = date.getUTCFullYear();

//Ponemos el nombre del archivo
let nombreArchivo = "todosUsuarios" + fechaDia + "_" + fechaMes + "_" + fechaAño + ".";
var ws = wb.addWorksheet(nombreArchivo);

//Creamos la ruta del excel
const path = require('path');
const pathExcel = path.join(__dirname, 'excel', nombreArchivo + '.xlsx');

//Escribir o guardar
wb.write(pathExcel, function(err, stats){
  if(err) console.log(err);
});