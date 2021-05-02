using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using MathNet.Numerics.LinearAlgebra;
using System;

using Random = UnityEngine.Random;

public class NNet : MonoBehaviour
{
    public Matrix<float> inputLayer = Matrix<float>.Build.Dense(1, 3); //3 sensörden oluşan tek boyutlu input matrisi

    public List<Matrix<float>> hiddenLayers = new List<Matrix<float>>();//n sayıda hidden layer oluşturabilmek için hidden layer değişkeni

    public Matrix<float> outputLayer = Matrix<float>.Build.Dense(1, 2);//hızlanma ve dönüş açısından oluşan tek boyutlu output matrisi

    public List<Matrix<float>> weights = new List<Matrix<float>>();

    public List<float> biases = new List<float>();

    public float fitness;

    public void Initialise (int hiddenLayerCount, int hiddenNeuronCount)
    {
        inputLayer.Clear();
        hiddenLayers.Clear();
        outputLayer.Clear();
        weights.Clear();
        biases.Clear();

        for (int i = 0; i < hiddenLayerCount + 1; i++)//hidden layer katmanlarını oluşturma
        {

            Matrix<float> f = Matrix<float>.Build.Dense(1, hiddenNeuronCount);//hidden layerların içerisindeki nöron sayısı

            hiddenLayers.Add(f);

            biases.Add(Random.Range(-1f, 1f));//1 ve -1 arasında rastgele bias eklenmesi

            //WEIGHTS
            if (i == 0)//input layer ve hidden layer arasındaki nöron sayısı farkından dolayı ayrı ağırlık eklememiz gerek
            {
                Matrix<float> inputToH1 = Matrix<float>.Build.Dense(3, hiddenNeuronCount);
                weights.Add(inputToH1);
            }
            //hidden layerlar arası ağrlık oluşturulması
            Matrix<float> HiddenToHidden = Matrix<float>.Build.Dense(hiddenNeuronCount, hiddenNeuronCount);
            weights.Add(HiddenToHidden);

        }
        //hidden layer ve output layer arasındaki bias ve ağırlık değerleri
        Matrix<float> OutputWeight = Matrix<float>.Build.Dense(hiddenNeuronCount, 2);
        weights.Add(OutputWeight);
        biases.Add(Random.Range(-1f, 1f));

        //atanan ağırlıkların -1 ve 1 arasında rastgele seçilmesi
        for (int i = 0; i < weights.Count; i++)
        {

            for (int x = 0; x < weights[i].RowCount; x++)
            {

                for (int y = 0; y < weights[i].ColumnCount; y++)
                {

                    weights[i][x, y] = Random.Range(-1f, 1f);

                }

            }

        }

    }

    public NNet InitialiseCopy (int hiddenLayerCount, int hiddenNeuronCount) //hidden layer ve nöronları kopyalama işlemi
    {
        NNet n = new NNet();

        List<Matrix<float>> newWeights = new List<Matrix<float>>();

        for (int i = 0; i < this.weights.Count; i++)
        {
            Matrix<float> currentWeight = Matrix<float>.Build.Dense(weights[i].RowCount, weights[i].ColumnCount);

            for (int x = 0; x < currentWeight.RowCount; x++)
            {
                for (int y = 0; y < currentWeight.ColumnCount; y++)
                {
                    currentWeight[x, y] = weights[i][x, y];
                }
            }

            newWeights.Add(currentWeight);
        }

        List<float> newBiases = new List<float>();

        newBiases.AddRange(biases);

        n.weights = newWeights;
        n.biases = newBiases;

        n.InitialiseHidden(hiddenLayerCount, hiddenNeuronCount);

        return n;
    }

    public void InitialiseHidden (int hiddenLayerCount, int hiddenNeuronCount)
    {
        inputLayer.Clear();
        hiddenLayers.Clear();
        outputLayer.Clear();

        for (int i = 0; i < hiddenLayerCount + 1; i ++)
        {
            Matrix<float> newHiddenLayer = Matrix<float>.Build.Dense(1, hiddenNeuronCount);
            hiddenLayers.Add(newHiddenLayer);
        }

    }
    public (float, float) RunNetwork (float a, float b, float c)
    {
        inputLayer[0, 0] = a;
        inputLayer[0, 1] = b;
        inputLayer[0, 2] = c;   //sensörlerin seçilmesi

        inputLayer = inputLayer.PointwiseTanh();//değerlerin -1 ile 1 arasına alınması 

        hiddenLayers[0] = ((inputLayer * weights[0]) + biases[0]).PointwiseTanh();//ilk hidden layerin aktive edilmesi

        for (int i = 1; i < hiddenLayers.Count; i++)
        {
            hiddenLayers[i] = ((hiddenLayers[i - 1] * weights[i]) + biases[i]).PointwiseTanh();
        }
        //varsa diğer hidden layerlerin hesaplanması

        outputLayer = ((hiddenLayers[hiddenLayers.Count-1]*weights[weights.Count-1])+biases[biases.Count-1]).PointwiseTanh();
        //son hidden layer ile outpu layer arasındaki hesapların yapılması

        return (Sigmoid(outputLayer[0,0]), (float)Math.Tanh(outputLayer[0,1]));
        //dönüş açısını normal gönderirken hızlanmayı sigmoid fonksiyona sokuyoruz bu sayede 0 ile 1 arasında oluyoe
    }

    private float Sigmoid (float s)
    {
        return (1 / (1 + Mathf.Exp(-s)));
    }

}
