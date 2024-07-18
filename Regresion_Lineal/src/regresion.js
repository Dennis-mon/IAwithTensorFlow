const tf = require('@tensorflow/tfjs'); // TensorFlow

//Calse encargada de toda la lógia de regresión lineal
class Regresion{

    //Features = , Labels =
    constructor(features, labels){

        //Declaramos los valores independientes (features) y los valores dependientes (labels)
        this.features = tf.tensor(features);
        this.labels = tf.tensor(labels);

        // Declaramos los valores de 'm' y 'b' que queremos encontrar para nuestra ecuación
        this.w = tf.variable(tf.scalar(Math.random()));
        this.b = tf.variable(tf.scalar(Math.random()));

        this.train = this.train.bind(this);

    }

    test( ){
        return this.w.mul(this.features).add(this.b);
    }

    loss(y_hat) {
        return y_hat.sub(this.labels).square().mean();
    }

    //Entrenamiento de la IA el número de veces que tenga la variable 'iterations' asociado
    train(){
        for(let i = 0; i < 2000; i++){

            let optimizer = tf.train.sgd(0.05);

            optimizer.minimize(() => {
                let y_hat =  this.test();
                let stepLoss = this.loss(y_hat);
                return stepLoss;
            });
        }
    }

    predict( ){

        this.w.print();
        this.b.print();
        return this.w.mul(feature).add(this.b);
    }

}


//Exoprtamos la clase para poder usarla desde fuera
module.exports = Regresion;