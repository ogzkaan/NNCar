using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(NNet))]
public class CarController : MonoBehaviour
{
    private Vector3 startPosition, startRotation;
    private NNet network;

    [Range(-1f,1f)]
    public float a,t;      //hızlanma ve dnöüş açısı

    public float timeSinceStart = 0f;   //Geçen süre

       //fitness ayarları
    public float overallFitness;
    public float distanceMultipler = 1.4f;
    public float avgSpeedMultiplier = 0.2f;
    public float sensorMultiplier = 0.1f;

    
    public int LAYERS = 1;
    public int NEURONS = 10;

    private Vector3 lastPosition;
    private float totalDistanceTravelled;
    private float avgSpeed;

    private float aSensor,bSensor,cSensor;

    private void Awake() {                      //sahne yüklendiğinde aracın ve network bilgileriin alınması
        startPosition = transform.position;
        startRotation = transform.eulerAngles;
        network = GetComponent<NNet>();
    }

    public void ResetWithNetwork (NNet net)//Sinir ağını gönderme
    {
        network = net;
        Reset();
    }

    

    public void Reset()   
    { 

        timeSinceStart = 0f;
        totalDistanceTravelled = 0f;
        avgSpeed = 0f;
        lastPosition = startPosition;
        overallFitness = 0f;
        transform.position = startPosition;
        transform.eulerAngles = startRotation;
    }

    private void OnCollisionEnter (Collision collision) 
    {
        print(overallFitness);
        Death();
       
    }

    private void FixedUpdate() {

        InputSensors();
        lastPosition = transform.position;

        (a, t) = network.RunNetwork(aSensor, bSensor, cSensor);

        MoveCar(a,t);

        timeSinceStart += Time.deltaTime;

        CalculateFitness();

    }
    private Vector3 inp;
    public void MoveCar(float v, float h)
    {
        inp = Vector3.Lerp(Vector3.zero, new Vector3(0, 0, v * 11.4f), 0.02f); //kademeli olarak hız arrtırma
        inp = transform.TransformDirection(inp);
        transform.position += inp;

        transform.eulerAngles += new Vector3(0, (h * 90) * 0.02f, 0);   //aracın kademeli olarak dönmesi için dönüş açısını yavaşça arrtırma
    }

    private void Death () //araç çarptığında network bilgilerinin Genetic Managera gönderilmesi
    {
        GameObject.FindObjectOfType<GeneticManager>().Death(overallFitness, network);
    }

    private void CalculateFitness() //fitness hesaplama
    {

        totalDistanceTravelled += Vector3.Distance(transform.position,lastPosition);
        avgSpeed = totalDistanceTravelled/timeSinceStart;

        overallFitness = (totalDistanceTravelled*distanceMultipler)+(avgSpeed*avgSpeedMultiplier)+(((aSensor+bSensor+cSensor)/3)*sensorMultiplier);

        if (timeSinceStart > 20 && overallFitness < 40) {
            Death();
        }
        
        if (overallFitness >= 1500) {
            Death();
        }

    }

    private void InputSensors() 
    {

        Vector3 a = (transform.forward+transform.right);     //aracın merkezinden çıkan 3 adet ışın
        Vector3 b = (transform.forward);
        Vector3 c = (transform.forward-transform.right);

        Ray r = new Ray(transform.position,a);
        RaycastHit hit;

        if (Physics.Raycast(r, out hit)) {                    //ışınların çoğunlıkla 1 ve sıfır arasında olması için uzaklık değerinin 20 ile bölüyorum
            aSensor = hit.distance/20;                        //basit bir linneralizasyon olarak düşünebiliriz
            Debug.DrawLine(r.origin, hit.point, Color.blue);
        }

        r.direction = b;

        if (Physics.Raycast(r, out hit)) {
            bSensor = hit.distance/20;
            Debug.DrawLine(r.origin, hit.point, Color.blue);
        }

        r.direction = c;

        if (Physics.Raycast(r, out hit)) {
            cSensor = hit.distance/20;
            Debug.DrawLine(r.origin, hit.point, Color.blue);
        }

    }

    

}
