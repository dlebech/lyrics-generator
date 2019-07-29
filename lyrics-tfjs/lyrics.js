// I should probably have used create react app for this... oh well :-)
((tf, SJS) => {
  const sleep = ms => {
    return new Promise(resolve => {
      setTimeout(resolve, ms);
    });
  };

  const softmaxSample = async (t, randomness) => {
    if (randomness === 0) return t.argMax().data();

    t = t.log().div(tf.tensor(randomness));
    t = t.exp().div(t.exp().sum());
    const probabilities = await t.data();
    const draw = SJS.Multinomial(1, probabilities).draw();
    return tf.tensor(draw).argMax().data();
  };

  const createElementLogger = elementId => {
    return txt => {
      document.getElementById(elementId).textContent = txt;
    };
  };

  class LyricsGenerator {
    constructor() {
      // this.debug = true;
    }

    async prepareGenerator(logger) {
      logger('Loading Tensorflow model...');
      this.model = await tf.loadLayersModel('data/model.json');
      this.padTo = this.model.inputs[0].shape[1];

      logger('Loading word mappings...');
      this.words = await fetch('data/words.json')
        .then(res => res.json());

      logger('Creating an inverse word lookup table...');
      this.reverseWords = Object.keys(this.words).reduce((obj, word) => {
        const index = this.words[word];
        obj[index] = word;
        return obj;
      }, {});
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

    async createLyric(textSeed, textLength, randomness, logger) {
      randomness = parseFloat(randomness);
      if (!this.words || !this.model) return '';

      let textOutput = textSeed;
      logger('Creating lyrics...');
      await sleep(20);
      console.log(`Generating lyric from "${textSeed}" with randomness ${randomness}...`);

      let lastLog = new Date();

      while (textOutput.length < textLength) {
        const tokenVector = this.textToVec(textOutput);
        if (this.debug) console.log('Token vector', tokenVector);

        const paddedTensor = this
          .padVector(tokenVector, this.padTo)
          .reshape([1, this.padTo]);
        if (this.debug) paddedTensor.print();

        const prediction = await this.model.predict(paddedTensor);
        if (this.debug) prediction.print()

        // The prediction is a 2D of potentially multiple predictions.
        // Squeeze removes one of the dimensions to make it nicer to work with :-)
        const index = await softmaxSample(prediction.squeeze(), randomness);
        let word = this.reverseWords[index[0]];
        if (!word) {
          console.log('Found empty word, curious');
          // Avoid printing "undefined"
          word = '';
        }
        textOutput += ' ' + word;

        // Log the progress every 3 seconds
        if ((new Date() - lastLog) >= 3000) {
          logger(`${textOutput.length} / ${textLength} characters created...`);
          lastLog = new Date();
          await sleep(100);
        }
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
        generator.createLyric(
          textSeed.value,
          textLength.value,
          randomness.value,
          createElementLogger('generate-status')
        )
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
        generator.prepareGenerator(createElementLogger('status'))
          .then(() => prepareForm(generator));
      }, 50);
    };
  };

  window.onload = () => {
    prepareIntro();
  };
})(window.tf, window.SJS);
