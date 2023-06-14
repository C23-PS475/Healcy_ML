int analogApin = 4;
float analogA;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  pinMode(analogApin, INPUT);
}
  
void loop() {
  // put your main code here, to run repeatedly:
  analogA = analogRead(analogApin) / 113.0; // Konversi dari 0-1023 menjadi 0-9
  analogA /= 1000.0; // Konversi dari 0-9 menjadi 0.001-0.01
  
  Serial.print("Kontraksi = ");
  Serial.print(analogA, 4); // Menampilkan dengan 4 angka di belakang koma
  Serial.println();
  
  delay(1000);
}
