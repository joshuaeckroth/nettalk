(ns nettalk.core
  (:import (org.encog Encog))
  (:import (org.encog.engine.network.activation ActivationSigmoid))
  (:import (org.encog.mathutil.randomize ConsistentRandomizer))
  (:import (org.encog.ml.data.basic BasicMLDataSet))
  (:import (org.encog.neural.networks BasicNetwork))
  (:import (org.encog.neural.networks.layers BasicLayer))
  (:import (org.encog.neural.networks.training.propagation.back Backpropagation))
  (:require [clojure.java.io :as io])
  (:require [clojure.string :as str]))

(defn convert-char
  [c]
  (if (= \space c) (concat [1] (repeat 26 0))
      (let [n (- (int c) (int \a))]
        (concat (repeat (inc n) 0)
                [1]
                (repeat (dec (- 26 n)) 0)))))

(defn convert-phoneme
  [phonemes p]
  (if (= \space p) (concat [1] (repeat (count phonemes) 0))
      (let [c (count phonemes)
            n (first (filter (fn [i] (= p (nth phonemes i))) (range c)))]
        (concat (repeat (inc n) 0)
                [1]
                (repeat (dec (- c n)) 0)))))

(defn convert-to-unary
  [text goal phonemes]
  (let [text-vec (vec text)
        goal-vec (vec goal)]
    {:input (for [i (range 3 (- (count text-vec) 4))]
              (map #(double-array (convert-char %)) (subvec text-vec (- i 3) (+ i 5))))
     :ideal (for [i (range 3 (- (count text-vec) 4))]
              (map #(double-array (convert-phoneme phonemes %)) (subvec goal-vec (- i 3) (+ i 5))))}))

(defn read-text
  [fname db]
  (with-open [rdr (io/reader fname)]
    (doall (for [line (line-seq rdr)]
             (let [words (-> line
                            (str/lower-case)
                            (str/replace #"[^a-z]" " ")
                            (str/replace #"^\s+" "")
                            (str/replace #"\s+$" "")
                            (str/replace #"\s+" " ")
                            (str/split #"\s"))]
               (for [word words]
                 (if (get db word) word (apply str (repeat (count word) \space)))))))))

(defn assoc-text-goal
  [lines-words db]
  (for [words lines-words]
    (for [word words]
      (get db word (apply str (repeat (count word) \space))))))

(defn render-text
  [lines-words]
  (for [words lines-words]
    (format "   %s   " (str/join " " words))))

(defn read-db
  []
  (with-open [rdr (io/reader "nettalk.data")]
    (doall (into {} (for [line (drop 10 (line-seq rdr))]
                      (let [[word phoneme _ _] (str/split line #"\t")]
                        [word phoneme]))))))

(defn generate-dataset
  []
  (let [db (read-db)
        phonemes (sort (set (apply concat (vals db))))
        lines-words (read-text "moby-dick-short.txt" db)
        goals (vec (render-text (assoc-text-goal lines-words db)))
        text (vec (render-text lines-words))
        unary (for [i (range (count text))]
                (convert-to-unary (nth text i) (nth goals i) phonemes))
        input (into-array (map double-array (apply concat (mapcat :input unary))))
        ideal (into-array (map double-array (apply concat (mapcat :ideal unary))))]
    (BasicMLDataSet. input ideal)))

(defn generate-network
  []
  (let [net (doto (BasicNetwork.)
              ;; input layer
              (.addLayer (BasicLayer. nil true 27))
              ;; hidden layer
              (.addLayer (BasicLayer. (ActivationSigmoid.) true 80))
              ;; output layer
              (.addLayer (BasicLayer. (ActivationSigmoid.) false 52))
              (.. (getStructure) (finalizeStructure))
              (.reset))]
    (.randomize (ConsistentRandomizer. -1 1 500) net)
    net))

(defn train
  []
  (let [net (generate-network)
        dataset (generate-dataset)
        train (Backpropagation. net dataset 0.01 0.1)]
    (loop [i 0]
      (if (not= i 1000)
        (do (.iteration train)
            (println (.getError train))
            (recur (inc i)))))))

(defn -main
  []
  (train))


