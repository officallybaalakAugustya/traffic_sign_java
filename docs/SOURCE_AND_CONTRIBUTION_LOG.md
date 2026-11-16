# Source, Contribution, and Academic Integrity Log

**Project:** Traffic Sign Recognition and Road Safety Monitor  
**Course:** Deep Learning (CSAI3027P) — PBL & Hackathon  
**Student Declaration:** Individual Project  

This document accounts for all external scientific literature, datasets, open-source libraries, architectural foundations, and permitted AI assistance utilized in this project, ensuring complete transparency and academic integrity.

---

## 1. Academic Literature & Theoretical Foundations

1. **McCulloch, W. S., & Pitts, W. (1943).** *A logical calculus of the ideas immanent in nervous activity.* The Bulletin of Mathematical Biophysics, 5(4), 115-133.
   * *Usage in Project:* Theoretical foundation for `McCullochPittsNeuron` logic unit.
2. **Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986).** *Learning representations by back-propagating errors.* Nature, 323(6088), 533-536.
   * *Usage in Project:* Derivation of analytical gradients in `MLPNetwork` backpropagation.
3. **Stallkamp, J., Schlipsing, M., Salmen, J., & Igel, C. (2012).** *Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition.* Neural Networks, 32, 323-332.
   * *Usage in Project:* GTSRB dataset definition, classification benchmark standards, and class taxonomy.
4. **Houben, S., Stallkamp, J., Salmen, J., Schlipsing, M., & Igel, C. (2013).** *Detection of traffic signs in real-world images: The German Traffic Sign Detection Benchmark.* IJCNN 2013.
   * *Usage in Project:* GTSDB full-scene dataset and detection evaluation methodology.
5. **Nesterov, Y. (1983).** *A method for solving the convex programming problem with convergence rate $O(1/k^2)$.* Soviet Mathematics Doklady.
   * *Usage in Project:* Formulation for `NesterovOptimizer` look-ahead momentum.

---

## 2. Dataset Attribution & Licenses

* **GTSRB (German Traffic Sign Recognition Benchmark):**
  * *Provider:* Institut für Neuroinformatik, Ruhr-Universität Bochum.
  * *License:* Free for academic and research evaluation.
  * *Permitted Scope:* Benchmark evaluation of multi-class traffic sign classifiers.
* **GTSDB (German Traffic Sign Detection Benchmark):**
  * *Provider:* Institut für Neuroinformatik, Ruhr-Universität Bochum.
  * *License:* Free for academic and research evaluation.

---

## 3. Third-Party Libraries & Software Dependencies

All direct third-party dependencies are declared in `pom.xml`:
* **JUnit 5 (org.junit.jupiter):** Eclipse Public License 2.0. Used for unit testing.
* **AssertJ (org.assertj:assertj-core):** Apache License 2.0. Used for test assertions.
* **SLF4J & Logback:** MIT & EPL. Used for logging.
* **Jackson Databind (com.fasterxml.jackson.core):** Apache License 2.0. Used for JSON manifest serialization.

*All core mathematical operations, neuron models, backpropagation routines, and evaluation metrics are written in pure Java without third-party mathematical black boxes.*

---

## 4. Generative AI Assistance Disclosure

* **Role of AI Assistant (Antigravity):**
  * Assisted in initial project scaffolding, documentation structuring, Maven configuration, and technical drafting.
  * *Student Responsibility:* The student is responsible for understanding, verifying, and independently modifying all architectural components, explaining mathematical formulations during the viva, and running all tests.
