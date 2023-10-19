# Modelle und Ebenen

Beim maschinellen Lernen ist ein *Modell* eine Funktion mit *lernbaren* [Parametern](https://developers.google.com/machine-learning/glossary/#parameter) , die eine Eingabe einer Ausgabe zuordnet. Die optimalen Parameter werden durch Training des Modells anhand von Daten ermittelt. Ein gut trainiertes Modell liefert eine genaue Zuordnung von der Eingabe zur gewünschten Ausgabe.

In TensorFlow.js gibt es zwei Möglichkeiten, ein Modell für maschinelles Lernen zu erstellen:

1. Verwenden der Layers-API, bei der Sie ein Modell mithilfe von *Layern* erstellen.
2. Verwenden der Core-API mit Operationen auf niedrigerer Ebene wie `tf.matMul()` , `tf.add()` usw.

Zunächst werfen wir einen Blick auf die Layers-API, eine übergeordnete API zum Erstellen von Modellen. Anschließend zeigen wir, wie Sie dasselbe Modell mithilfe der Core-API erstellen.

## Erstellen von Modellen mit der Layers-API

There are two ways to create a model using the Layers API: A *sequential* model, and a *functional* model. The next two sections look at each type more closely.

### Das sequentielle Modell

The most common type of model is the <code>[Sequential](https://js.tensorflow.org/api/0.15.1/#class:Sequential)</code> model, which is a linear stack of layers. You can create a <code>Sequential</code> model by passing a list of layers to the <code>[sequential()](https://js.tensorflow.org/api/0.15.1/#sequential)</code> function:

```js
const model = tf.sequential({
 layers: [
   tf.layers.dense({inputShape: [784], units: 32, activation: 'relu'}),
   tf.layers.dense({units: 10, activation: 'softmax'}),
 ]
});
```

Oder über die `add()` -Methode:

```js
const model = tf.sequential();
model.add(tf.layers.dense({inputShape: [784], units: 32, activation: 'relu'}));
model.add(tf.layers.dense({units: 10, activation: 'softmax'}));
```

> WICHTIG: Die erste Ebene im Modell benötigt eine `inputShape` . Stellen Sie sicher, dass Sie die Stapelgröße ausschließen, wenn Sie `inputShape` bereitstellen. Wenn Sie beispielsweise vorhaben, die Modelltensoren der Form `[B, 784]` zu füttern, wobei `B` eine beliebige Stapelgröße sein kann, geben Sie beim Erstellen des Modells `inputShape` als `[784]` an.

Sie können über `model.layers` auf die Ebenen des Modells zugreifen, genauer gesagt über `model.inputLayers` und `model.outputLayers` .

### Das Funktionsmodell

Eine andere Möglichkeit, ein `LayersModel` zu erstellen, ist die Funktion `tf.model()` . Der Hauptunterschied zwischen `tf.model()` und `tf.sequential()` besteht darin, dass Sie `tf.model()` ein beliebiges Diagramm von Ebenen erstellen können, sofern diese keine Zyklen haben.

Hier ist ein Codeausschnitt, der dasselbe Modell wie oben mithilfe der `tf.model()` -API definiert:

```js
// Create an arbitrary graph of layers, by connecting them
// via the apply() method.
const input = tf.input({shape: [784]});
const dense1 = tf.layers.dense({units: 32, activation: 'relu'}).apply(input);
const dense2 = tf.layers.dense({units: 10, activation: 'softmax'}).apply(dense1);
const model = tf.model({inputs: input, outputs: dense2});
```

Wir rufen `apply()` auf jeder Ebene auf, um sie mit der Ausgabe einer anderen Ebene zu verbinden. Das Ergebnis von `apply()` ist in diesem Fall ein `SymbolicTensor` , der sich wie ein `Tensor` verhält, jedoch keine konkreten Werte aufweist.

Beachten Sie, dass wir im Gegensatz zum sequentiellen Modell einen `SymbolicTensor` über `tf.input()` erstellen, anstatt der ersten Ebene eine `inputShape` bereitzustellen.

`apply()` kann Ihnen auch einen konkreten `Tensor` liefern, wenn Sie ihm einen konkreten `Tensor` übergeben:

```js
const t = tf.tensor([-2, 1, 0, 5]);
const o = tf.layers.activation({activation: 'relu'}).apply(t);
o.print(); // [0, 1, 0, 5]
```

Dies kann nützlich sein, wenn Sie Ebenen isoliert testen und ihre Ausgabe sehen möchten.

Genau wie in einem sequentiellen Modell können Sie über `model.layers` und insbesondere `model.inputLayers` und `model.outputLayers` auf die Ebenen des Modells zugreifen.

## Validierung

Sowohl das sequentielle Modell als auch das Funktionsmodell sind Instanzen der `LayersModel` Klasse. Einer der Hauptvorteile der Arbeit mit einem `LayersModel` ist die Validierung: Sie werden gezwungen, die Eingabeform anzugeben, und werden diese später zur Validierung Ihrer Eingabe verwenden. Das `LayersModel` führt auch eine automatische Forminferenz durch, während die Daten durch die Ebenen fließen. Wenn Sie die Form im Voraus kennen, kann das Modell seine Parameter automatisch erstellen und Ihnen mitteilen, ob zwei aufeinanderfolgende Schichten nicht miteinander kompatibel sind.

## Modellzusammenfassung

Rufen Sie `model.summary()` auf, um eine nützliche Zusammenfassung des Modells zu drucken, die Folgendes enthält:

- Name und Typ aller Ebenen im Modell.
- Ausgabeform für jede Ebene.
- Anzahl der Gewichtsparameter jeder Schicht.
- Wenn das Modell eine allgemeine Topologie hat (siehe unten), erhält jede Schicht die Eingaben
- Die Gesamtzahl der trainierbaren und nicht trainierbaren Parameter des Modells.

Für das oben definierte Modell erhalten wir die folgende Ausgabe auf der Konsole:

<table>
  <tr>
   <td>Schicht (Typ)    </td>
   <td>Ausgabeform    </td>
   <td>Parameter #    </td>
  </tr>
  <tr>
   <td>dicht_Dense1 (Dicht)    </td>
   <td>[null,32]    </td>
   <td>25120    </td>
  </tr>
  <tr>
   <td>dicht_Dense2 (Dicht)    </td>
   <td>[null,10]    </td>
   <td>330    </td>
  </tr>
  <tr>
   <td colspan="3">Gesamtparameter: 25450<br> Trainierbare Parameter: 25450<br> Nicht trainierbare Parameter: 0    </td>
  </tr>
</table>

Beachten Sie die `null` in den Ausgabeformen der Ebenen: eine Erinnerung daran, dass das Modell erwartet, dass die Eingabe eine Stapelgröße als äußerste Dimension hat, die in diesem Fall aufgrund des `null` flexibel sein kann.

## Serialisierung

Einer der Hauptvorteile der Verwendung eines `LayersModel` gegenüber der API auf niedrigerer Ebene ist die Möglichkeit, ein Modell zu speichern und zu laden. Ein `LayersModel` kennt Folgendes:

- die Architektur des Modells, sodass Sie das Modell neu erstellen können.
- die Gewichte des Modells
- die Trainingskonfiguration (Verlust, Optimierer, Metriken).
- den Status des Optimierers, sodass Sie das Training fortsetzen können.

Zum Speichern oder Laden eines Modells ist nur eine Codezeile erforderlich:

```js
const saveResult = await model.save('localstorage://my-model-1');
const model = await tf.loadLayersModel('localstorage://my-model-1');
```

Das obige Beispiel speichert das Modell im lokalen Speicher im Browser. <code>[model.save() documentation](https://js.tensorflow.org/api/latest/#tf.Model.save)</code> zum [Speichern](save_load.md) auf verschiedenen Medien (z. B. Dateispeicher, <code>IndexedDB</code> , Auslösen eines Browser-Downloads usw.)

## Benutzerdefinierte Ebenen

Ebenen sind die Bausteine ​​eines Modells. Wenn Ihr Modell eine benutzerdefinierte Berechnung durchführt, können Sie eine benutzerdefinierte Ebene definieren, die gut mit den übrigen Ebenen interagiert. Nachfolgend definieren wir eine benutzerdefinierte Ebene, die die Summe der Quadrate berechnet:

```js
class SquaredSumLayer extends tf.layers.Layer {
 constructor() {
   super({});
 }
 // In this case, the output is a scalar.
 computeOutputShape(inputShape) { return []; }

 // call() is where we do the computation.
 call(input, kwargs) { return input.square().sum();}

 // Every layer needs a unique name.
 getClassName() { return 'SquaredSum'; }
}
```

Um es zu testen, können wir die Methode `apply()` mit einem konkreten Tensor aufrufen:

```js
const t = tf.tensor([-2, 1, 0, 5]);
const o = new SquaredSumLayer().apply(t);
o.print(); // prints 30
```

> WICHTIG: Wenn Sie eine benutzerdefinierte Ebene hinzufügen, verlieren Sie die Möglichkeit, ein Modell zu serialisieren.

## Erstellen von Modellen mit der Core API

Am Anfang dieses Leitfadens haben wir erwähnt, dass es zwei Möglichkeiten gibt, ein Modell für maschinelles Lernen in TensorFlow.js zu erstellen.

Als allgemeine Faustregel gilt, dass Sie immer zuerst versuchen sollten, die Layers-API zu verwenden, da sie der bewährten Keras-API nachempfunden ist, die [Best Practices befolgt und die kognitive Belastung reduziert](https://keras.io/why-use-keras/) . Die Layers API bietet auch verschiedene Standardlösungen wie Gewichtsinitialisierung, Modellserialisierung, Überwachungstraining, Portabilität und Sicherheitsprüfung.

Möglicherweise möchten Sie die Core-API immer dann verwenden, wenn:

- Sie benötigen maximale Flexibilität oder Kontrolle.
- Sie benötigen keine Serialisierung oder können Ihre eigene Serialisierungslogik implementieren.

Modelle in der Core-API sind lediglich Funktionen, die einen oder mehrere `Tensors` annehmen und einen `Tensor` zurückgeben. Das gleiche Modell wie oben, das mit der Core-API geschrieben wurde, sieht folgendermaßen aus:

```js
// The weights and biases for the two dense layers.
const w1 = tf.variable(tf.randomNormal([784, 32]));
const b1 = tf.variable(tf.randomNormal([32]));
const w2 = tf.variable(tf.randomNormal([32, 10]));
const b2 = tf.variable(tf.randomNormal([10]));

function model(x) {
  return x.matMul(w1).add(b1).relu().matMul(w2).add(b2).softmax();
}
```

Beachten Sie, dass wir in der Core-API für die Erstellung und Initialisierung der Gewichte des Modells verantwortlich sind. Jede Gewichtung wird durch eine `Variable` unterstützt, die TensorFlow.js signalisiert, dass diese Tensoren lernbar sind. Sie können eine `Variable` mit [tf.variable()](https://js.tensorflow.org/api/latest/#variable) erstellen und einen vorhandenen `Tensor` übergeben.

In diesem Leitfaden haben Sie sich mit den verschiedenen Möglichkeiten zum Erstellen eines Modells mithilfe der Layers und der Core API vertraut gemacht. Sehen Sie sich als Nächstes die Anleitung zum [Trainieren von Modellen](train_models.md) an, um zu erfahren, wie Sie ein Modell trainieren.
