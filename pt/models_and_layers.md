# Modelos e camadas

No aprendizado de máquina, um *modelo* é uma função com [parâmetros](https://developers.google.com/machine-learning/glossary/#parameter) que *podem ser aprendidos* que mapeia uma entrada para uma saída. Os parâmetros ideais são obtidos treinando o modelo nos dados. Um modelo bem treinado fornecerá um mapeamento preciso da entrada até a saída desejada.

No TensorFlow.js existem duas maneiras de criar um modelo de aprendizado de máquina:

1. usando a API Layers onde você constrói um modelo usando *camadas* .
2. usando a API Core com operações de nível inferior, como `tf.matMul()` , `tf.add()` , etc.

Primeiro, veremos a API Layers, que é uma API de nível superior para construção de modelos. A seguir, mostraremos como construir o mesmo modelo usando a API Core.

## Criando modelos com a API Layers

Existem duas maneiras de criar um modelo usando a API Layers: um modelo *sequencial* e um modelo *funcional* . As próximas duas seções examinam cada tipo mais de perto.

### O modelo sequencial

O tipo mais comum de modelo é o modelo <code>[Sequential](https://js.tensorflow.org/api/0.15.1/#class:Sequential)</code> , que é uma pilha linear de camadas. Você pode criar um modelo <code>Sequential</code> passando uma lista de camadas para a função <code>[sequential()](https://js.tensorflow.org/api/0.15.1/#sequential)</code> :

```js
const model = tf.sequential({
 layers: [
   tf.layers.dense({inputShape: [784], units: 32, activation: 'relu'}),
   tf.layers.dense({units: 10, activation: 'softmax'}),
 ]
});
```

Ou através do método `add()` :

```js
const model = tf.sequential();
model.add(tf.layers.dense({inputShape: [784], units: 32, activation: 'relu'}));
model.add(tf.layers.dense({units: 10, activation: 'softmax'}));
```

> IMPORTANTE: A primeira camada do modelo precisa de um `inputShape` . Certifique-se de excluir o tamanho do lote ao fornecer `inputShape` . Por exemplo, se você planeja alimentar os tensores do modelo de forma `[B, 784]` , onde `B` pode ser qualquer tamanho de lote, especifique `inputShape` como `[784]` ao criar o modelo.

Você pode acessar as camadas do modelo via `model.layers` e, mais especificamente `model.inputLayers` e `model.outputLayers` .

### O modelo funcional

Outra forma de criar um `LayersModel` é através da função `tf.model()` . A principal diferença entre `tf.model()` e `tf.sequential()` é que `tf.model()` permite criar um gráfico arbitrário de camadas, desde que não tenham ciclos.

Aqui está um trecho de código que define o mesmo modelo acima usando a API `tf.model()` :

```js
// Create an arbitrary graph of layers, by connecting them
// via the apply() method.
const input = tf.input({shape: [784]});
const dense1 = tf.layers.dense({units: 32, activation: 'relu'}).apply(input);
const dense2 = tf.layers.dense({units: 10, activation: 'softmax'}).apply(dense1);
const model = tf.model({inputs: input, outputs: dense2});
```

Chamamos `apply()` em cada camada para conectá-la à saída de outra camada. O resultado de `apply()` neste caso é um `SymbolicTensor` , que atua como um `Tensor` , mas sem quaisquer valores concretos.

Observe que, diferentemente do modelo sequencial, criamos um `SymbolicTensor` via `tf.input()` em vez de fornecer um `inputShape` para a primeira camada.

`apply()` também pode fornecer um `Tensor` concreto, se você passar um `Tensor` concreto para ele:

```js
const t = tf.tensor([-2, 1, 0, 5]);
const o = tf.layers.activation({activation: 'relu'}).apply(t);
o.print(); // [0, 1, 0, 5]
```

Isso pode ser útil ao testar camadas isoladamente e ver sua saída.

Assim como em um modelo sequencial, você pode acessar as camadas do modelo via `model.layers` e, mais especificamente, `model.inputLayers` e `model.outputLayers` .

## Validação

Tanto o modelo sequencial quanto o modelo funcional são instâncias da classe `LayersModel` . Um dos principais benefícios de trabalhar com um `LayersModel` é a validação: ela força você a especificar o formato de entrada e o usará posteriormente para validar sua entrada. O `LayersModel` também faz inferência automática de formas à medida que os dados fluem pelas camadas. Conhecer a forma antecipadamente permite que o modelo crie automaticamente seus parâmetros e pode informar se duas camadas consecutivas não são compatíveis entre si.

## Resumo do modelo

Chame `model.summary()` para imprimir um resumo útil do modelo, que inclui:

- Nome e tipo de todas as camadas do modelo.
- Forma de saída para cada camada.
- Número de parâmetros de peso de cada camada.
- Se o modelo tiver topologia geral (discutida abaixo), as entradas que cada camada recebe
- O número total de parâmetros treináveis ​​e não treináveis ​​do modelo.

Para o modelo que definimos acima, obtemos a seguinte saída no console:

<table>
  <tr>
   <td>Camada (tipo)    </td>
   <td>Forma de saída    </td>
   <td>Parâmetro #    </td>
  </tr>
  <tr>
   <td>denso_Dense1 (denso)    </td>
   <td>[nulo,32]    </td>
   <td>25120    </td>
  </tr>
  <tr>
   <td>denso_Dense2 (denso)    </td>
   <td>[nulo,10]    </td>
   <td>330    </td>
  </tr>
  <tr>
   <td colspan="3">Parâmetros totais: 25450<br> Parâmetros treináveis: 25450<br> Parâmetros não treináveis: 0    </td>
  </tr>
</table>

Observe os valores `null` nas formas de saída das camadas: um lembrete de que o modelo espera que a entrada tenha um tamanho de lote como dimensão mais externa, que neste caso pode ser flexível devido ao valor `null` .

## Serialização

Um dos principais benefícios de usar um `LayersModel` em vez da API de nível inferior é a capacidade de salvar e carregar um modelo. Um `LayersModel` conhece:

- a arquitetura do modelo, permitindo recriar o modelo.
- os pesos do modelo
- a configuração do treinamento (perda, otimizador, métricas).
- o estado do otimizador, permitindo que você retome o treinamento.

Para salvar ou carregar um modelo basta apenas 1 linha de código:

```js
const saveResult = await model.save('localstorage://my-model-1');
const model = await tf.loadLayersModel('localstorage://my-model-1');
```

O exemplo acima salva o modelo no armazenamento local do navegador. Consulte a <code>[model.save() documentation](https://js.tensorflow.org/api/latest/#tf.Model.save)</code> e o guia [de salvamento e carregamento](save_load.md) para saber como salvar em diferentes mídias (por exemplo, armazenamento de arquivos, <code>IndexedDB</code> , acionar um download do navegador, etc.)

## Camadas personalizadas

Camadas são os blocos de construção de um modelo. Se o seu modelo estiver fazendo um cálculo personalizado, você poderá definir uma camada personalizada, que interage bem com o restante das camadas. Abaixo definimos uma camada personalizada que calcula a soma dos quadrados:

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

Para testá-lo, podemos chamar o método `apply()` com um tensor concreto:

```js
const t = tf.tensor([-2, 1, 0, 5]);
const o = new SquaredSumLayer().apply(t);
o.print(); // prints 30
```

> IMPORTANTE: Se você adicionar uma camada personalizada, perderá a capacidade de serializar um modelo.

## Criando modelos com a API Core

No início deste guia, mencionamos que existem duas maneiras de criar um modelo de aprendizado de máquina no TensorFlow.js.

A regra geral é sempre tentar usar a API Layers primeiro, uma vez que ela é modelada de acordo com a API Keras bem adotada, que segue [as melhores práticas e reduz a carga cognitiva](https://keras.io/why-use-keras/) . A API Layers também oferece várias soluções prontas para uso, como inicialização de peso, serialização de modelo, treinamento de monitoramento, portabilidade e verificação de segurança.

Você pode querer usar a API Core sempre que:

- Você precisa de flexibilidade ou controle máximo.
- Você não precisa de serialização ou pode implementar sua própria lógica de serialização.

Os modelos na API Core são apenas funções que pegam um ou mais `Tensors` e retornam um `Tensor` . O mesmo modelo acima escrito usando a API Core é assim:

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

Observe que na API Core somos responsáveis ​​por criar e inicializar os pesos do modelo. Cada peso é apoiado por uma `Variable` que sinaliza ao TensorFlow.js que esses tensores podem ser aprendidos. Você pode criar uma `Variable` usando [tf.variable()](https://js.tensorflow.org/api/latest/#variable) e passando um `Tensor` existente.

Neste guia você se familiarizou com as diferentes maneiras de criar um modelo usando as camadas e a API Core. A seguir, consulte o guia [de modelos de treinamento](train_models.md) para saber como treinar um modelo.
