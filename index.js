const tf = require('@tensorflow/tfjs-node');
const csv = require('fast-csv');
const _ = require('lodash');

const TRAIN_ROUND = 10;
const TIME_STEP = 3;
const LEARNING_RATE = 0.001;
const NUM_OUT = 0;

function readCSV() {
    return new Promise((resolve, reject) => {
        const arr = [];
        csv.parseFile('./csv/sanam.csv')
            .on('data', (data) => {
                arr.push(data);
            })
            .on('end', () => {
                resolve(arr);
            });
    })
}

async function getTouristCount(hour) {
    const result = [];
    const csvData = await readCSV();
    for (let i = csvData.length - 1; i >= 1 && hour > 0; i--) {
        const strData = csvData[i][0];
        const arrData = strData.split(';');
        for (let j = arrData.length - 1; j >= 1 && hour > 0; j--) {
            result.push(+arrData[j]);   //parseInt
            hour--;
        }
    }
    return result.reverse();
}

function prepareData(data) {
    //normalize
    const max = data.reduce((prev, val) => val > prev ? val : prev);
    const nomalizedData = data.map((val) => val / max);
    const range = _.range(TIME_STEP, nomalizedData.length - NUM_OUT);

    const xs = [], ys = [];

    range.forEach((i) => {
        let x = [];
        for (let j = i - TIME_STEP; j < i; j++) {
            x.push(nomalizedData[j]);
        }
        xs.push(x);
        ys.push(nomalizedData[i]);
    });

    return { xs, ys, max }
}

function createModel() {
    const model = tf.sequential(); // initial model

    //input
    model.add(tf.layers.lstm({
        units: 10,  // 50 node if more will slower
        inputShape: [TIME_STEP, 1], // input 3 time steps each time step 1 dimension  e.g. [10,20,30]
        returnSequences: false
    }));

    //output
    model.add(tf.layers.dense({
        units: 1,
        kernelInitializer: 'VarianceScaling',
        activation: 'relu'
    }));

    //optimize
    const optimizer = tf.train.adam(LEARNING_RATE);

    model.compile({
        optimizer,
        loss: 'meanSquaredError',
        metrics: ['accuracy']
    });

    return model;
}

async function trainModel(model, { xs, ys }) {
    let trainedXs = tf.tensor2d(xs);
    trainedXs = tf.reshape(trainedXs, [-1, TIME_STEP, 1]);

    let trainedYs = tf.tensor1d(ys);
    trainedYs = tf.reshape(trainedYs, [-1, 1]);

    await model.fit(trainedXs, trainedYs, {
        batchSize: 1,
        epochs: TRAIN_ROUND,
        shuffle: true,
        validationSplit: 0.2
    });

    return model;
}

function predict(model, input) {
    const r = model.predict(input);
    let result = r.dataSync()[0];
    return result;
}

function toTensor(dSet) {
    dSet = [dSet];
    dSet = tf.tensor2d(dSet);
    dSet = tf.reshape(dSet, [-1, TIME_STEP, 1]);
    return dSet;
}

async function main() {
    const counts = await getTouristCount(50);
    const input = counts.slice(counts.length - TIME_STEP - 5, counts.length - 5);
    const data = prepareData(counts);
    const model = createModel();
    const trainedModel = await trainModel(model, data);
    const predicted = predict(trainedModel, toTensor(input));
    console.log(input);
    console.log('predicted', Math.floor(predicted * data.max));
}

main();