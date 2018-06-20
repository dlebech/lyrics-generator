// I should probably have used create react app for this... oh well :-)
((tf, SJS) => {
  const softmaxSample = async (t, randomness) => {
    if (randomness === 0) return t.argMax().data();

    t = t.log().div(tf.tensor(randomness));
    t = t.exp().div(t.exp().sum());
    const probabilities = await t.data();
    const draw = SJS.Multinomial(1, probabilities).draw();
    return tf.tensor(draw).argMax().data();
  };

  const generateLogger = elementId => {
    return txt => {
      document.getElementById(elementId).textContent = txt;
    };
  };

  class LyricsGenerator {
    async prepareGenerator(logger) {
      logger('Loading Tensorflow model...');
      this.model = await tf.loadModel('data/model.json');

      logger('Loading word mappings...');
      this.words = await fetch('data/words.json')
        .then(res => res.json());

      logger('Creating an inverse word lookup table...');
      this.reverseWords = Object.keys(this.words).reduce((obj, word) => {
        const index = this.words[word];
        obj[index] = word;
        return obj;
      }, {});

      logger('Setting up Tensorflow network...');

      // Re-construct the network manually.
      // This is a hack because dropout and recurrent dropout has not been
      // implemented for GRU cells yet :-(
      // ... and I couldn't find a better way for now :-(
      const input = tf.layers.input({
        batchShape: this.model.getLayer('input_1').getConfig().batchInputShape,
      });
      const emb = tf.layers.embedding({
        inputDim: this.model.getLayer('embedding_1').getConfig().inputDim,
        outputDim: this.model.getLayer('embedding_1').getConfig().outputDim,
        maskZero: this.model.getLayer('embedding_1').getConfig().maskZero,
        inputLength: this.model.getLayer('embedding_1').getConfig().inputLength,
        weights: this.model.getLayer('embedding_1').getWeights()
      });
      const gru1 = tf.layers.gru({
        units: this.model.getLayer('gru_1').getConfig().units,
        returnSequences: this.model.getLayer('gru_1').getConfig().returnSequences,
        weights: this.model.getLayer('gru_1').getWeights()
      });
      const gru2 = tf.layers.gru({
        units: this.model.getLayer('gru_2').getConfig().units,
        returnSequences: this.model.getLayer('gru_2').getConfig().returnSequences,
        weights: this.model.getLayer('gru_2').getWeights()
      });
      const dense1 = tf.layers.dense({
        units: this.model.getLayer('dense_1').getConfig().units,
        activation: this.model.getLayer('dense_1').getConfig().activation,
        weights: this.model.getLayer('dense_1').getWeights()
      });
      const dropout = tf.layers.dropout({
        rate: this.model.getLayer('dropout_1').getConfig().rate
      });
      const dense2 = tf.layers.dense({
        units: this.model.getLayer('dense_2').getConfig().units,
        activation: this.model.getLayer('dense_2').getConfig().activation,
        weights: this.model.getLayer('dense_2').getWeights()
      });

      logger('Preparing Tensorflow model...');
      const output = dense2.apply(
        dropout.apply(
          dense1.apply(
            gru2.apply(
              gru1.apply(
                emb.apply(input)
              )
            )
          )
        )
      );

      this.newModel = tf.model({ inputs: input, outputs: output });
    }

    cleanText(text) {
      return text
        .replace(/[^\w\n\s]/g, '')
        .replace('\n', ' \n ');
    }

    textToVec(text) {
      text = this.cleanText(text);
      const tokens = text.split(' ');
      const tokenVector = [];
      let i = 0;
      while (i < tokens.length) {
        const newToken = this.words[tokens[i]];
        if (typeof newToken === 'undefined') {
          tokens.splice(i, 1);
          continue;
        }
        tokenVector.push(newToken);
        i++;
      }
      return tokenVector;
    }

    padVector(vector, length) {
      const t = tf.tensor(vector);
      return t.pad([[length - t.shape[0], 0]]);
    }

    async createLyric(textSeed, textLength, randomness) {
      randomness = parseFloat(randomness);
      if (!this.words || !this.model) return '';

      let textOutput = textSeed;
      console.log(`Generating lyric from "${textSeed}" with randomness ${randomness}...`);

      while (textOutput.length < textLength) {
        const tokenVector = this.textToVec(textOutput);
        if (this.debug) console.log('Token vector', tokenVector);

        const paddedTensor = this.padVector(tokenVector, 14).reshape([1, 14]); // TODO
        if (this.debug) paddedTensor.print();

        // TODO: Use loaded model instead.
        const prediction = await this.newModel.predict(paddedTensor);
        if (this.debug) prediction.print()

        // The prediction is a 2D of potentially multiple predictions.
        // Squeeze removes one of the dimensions to make it nicer to work with :-)
        const index = await softmaxSample(prediction.squeeze(), randomness);
        const word = this.reverseWords[index[0]];
        textOutput += ' ' + word;
      }

      return textOutput;
    }
  }

  const prepareForm = generator => {
    const form = document.getElementById('lyrics-form');
    const textSeed = document.getElementById('text-seed');
    const textLength = document.getElementById('text-length');
    const randomness = document.getElementById('randomness');
    const randomnessValue = document.getElementById('randomness-value');
    const output = document.getElementById('output');

    const mainLoader = document.getElementById('loader');
    const generateLoader = document.getElementById('generate-loader');

    // Setup form submit
    form.onsubmit = e => {
      e.preventDefault();

      output.textContent = '';
      generateLoader.style.display = '';

      // Wrap the lyrics call in a timeout to avoid block the UI while the loader is being shown.
      // Yes, this happens.
      setTimeout(() => {
        generator.createLyric(textSeed.value, textLength.value, randomness.value)
          .then(text => output.textContent = text)
          .then(() => generateLoader.style.display = 'none');
      }, 50);
    };

    // Set up randomness change
    randomness.oninput = () => {
      randomnessValue.textContent = randomness.value;
    };

    form.style.display = '';
    mainLoader.style.display = 'none';
  };

  const prepareIntro = () => {
    document.getElementById('load-button').onclick = () => {
      document.getElementById('intro-section').style.display = 'none';
      document.getElementById('main-section').style.display = '';
      setTimeout(() => {
        const generator = new LyricsGenerator();
        generator.prepareGenerator(generateLogger('status'))
          .then(() => prepareForm(generator));
      }, 50);
    };
  };

  window.onload = () => {
    prepareIntro();
  };
})(window.tf, window.SJS);
