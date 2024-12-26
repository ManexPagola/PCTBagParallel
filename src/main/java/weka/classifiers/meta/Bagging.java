/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *    Bagging.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.meta;

import java.util.*;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import weka.classifiers.Classifier;
import weka.classifiers.RandomizableParallelIteratedSingleClassifierEnhancer;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.*;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.matrix.DoubleVector;

/**
 <!-- globalinfo-start -->
 * Class for bagging a classifier to reduce variance. Can do classification and regression depending on the base learner. <br/>
 * <br/>
 * For more information, see<br/>
 * <br/>
 * Leo Breiman (1996). Bagging predictors. Machine Learning. 24(2):123-140.
 * <p/>
 <!-- globalinfo-end -->
 *
 <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;article{Breiman1996,
 *    author = {Leo Breiman},
 *    journal = {Machine Learning},
 *    number = {2},
 *    pages = {123-140},
 *    title = {Bagging predictors},
 *    volume = {24},
 *    year = {1996}
 * }
 * </pre>
 * <p/>
 <!-- technical-bibtex-end -->
 *
 <!-- options-start -->
 * Valid options are: <p/>
 * 
 * <pre> -P
 *  Size of each bag, as a percentage of the
 *  training set size. (default 100)</pre>
 * 
 * <pre> -O
 *  Calculate the out of bag error.</pre>
 *
 * <pre> -print
 *  Print the individual classifiers in the output</pre>
 *
 * <pre> -store-out-of-bag-predictions
 *  Whether to store out of bag predictions in internal evaluation object.</pre>
 *
 * <pre> -output-out-of-bag-complexity-statistics
 *  Whether to output complexity-based statistics when out-of-bag evaluation is performed.</pre>
 *
 * <pre> -represent-copies-using-weights
 *  Represent copies of instances using weights rather than explicitly.</pre>
 * 
 * <pre> -S &lt;num&gt;
 *  Random number seed.
 *  (default 1)</pre>
 * 
 * <pre> -num-slots &lt;num&gt;
 *  Number of execution slots.
 *  (default 1 - i.e. no parallelism)</pre>
 * 
 * <pre> -I &lt;num&gt;
 *  Number of iterations.
 *  (default 10)</pre>
 * 
 * <pre> -D
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console</pre>
 * 
 * <pre> -W
 *  Full name of base classifier.
 *  (default: weka.classifiers.trees.REPTree)</pre>
 * 
 * <pre> 
 * Options specific to classifier weka.classifiers.trees.REPTree:
 * </pre>
 * 
 * <pre> -M &lt;minimum number of instances&gt;
 *  Set minimum number of instances per leaf (default 2).</pre>
 * 
 * <pre> -V &lt;minimum variance for split&gt;
 *  Set minimum numeric class variance proportion
 *  of train variance for split (default 1e-3).</pre>
 * 
 * <pre> -N &lt;number of folds&gt;
 *  Number of folds for reduced error pruning (default 3).</pre>
 * 
 * <pre> -S &lt;seed&gt;
 *  Seed for random data shuffling (default 1).</pre>
 * 
 * <pre> -P
 *  No pruning.</pre>
 * 
 * <pre> -L
 *  Maximum tree depth (default -1, no maximum)</pre>
 * 
 * <pre> -I
 *  Initial class value count (default 0)</pre>
 * 
 * <pre> -R
 *  Spread initial count over all class values (i.e. don't use 1 per value)</pre>
 * 
 <!-- options-end -->
 *
 * Options after -- are passed to the designated classifier.<p>
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @author Len Trigg (len@reeltwo.com)
 * @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 * @version $Revision: 15801 $
 */
public class Bagging
  extends RandomizableParallelIteratedSingleClassifierEnhancer 
  implements WeightedInstancesHandler, AdditionalMeasureProducer,
             TechnicalInformationHandler, PartitionGenerator, Aggregateable<Bagging> {

  /** for serialization */
  static final long serialVersionUID = -115879962237199703L;
  
  /** The size of each bag sample, as a percentage of the training size */
  protected int m_BagSizePercent = 100;

  /** Whether to calculate the out of bag error */
  protected boolean m_CalcOutOfBag = false;

  /** Whether to represent copies of instances using weights rather than explicitly */
  protected boolean m_RepresentUsingWeights = false;

  /** The evaluation object holding the out of bag error, etc. */
  protected Evaluation m_OutOfBagEvaluationObject = null;

  /** Whether to store the out of bag predictions in the evaluation object. */
  private boolean m_StoreOutOfBagPredictions = false;

  /** Whether to output complexity-based statistics when OOB-evaluation is performed. */
  private boolean m_OutputOutOfBagComplexityStatistics;

  /** Whether class is numeric. */
  private boolean m_Numeric = false;

  /** Whether to print individual ensemble members in output.*/
  private boolean m_printClassifiers;

  /** Random number generator */
  protected Random m_random;

  /** Used to indicate whether an instance is in a bag or not */
  protected boolean[][] m_inBag;

  /** Reference to the training data */
  protected Instances m_data;
  
  /** Whether to classify using the Majority Voting combination rule. Can
   * assume that class is nominal. */
  private boolean m_MajorityVotingRule = false;

  /**
   * Constructor.
   */
  public Bagging() {
    
    m_Classifier = new weka.classifiers.trees.REPTree();
  }
  
  /**
   * Returns a string describing classifier
   * @return a description suitable for
   * displaying in the explorer/experimenter gui
   */
  public String globalInfo() {
 
    return "Class for bagging a classifier to reduce variance. Can do classification "
      + "and regression depending on the base learner. \n\n"
      + "For more information, see\n\n"
      + getTechnicalInformation().toString();
  }

  /**
   * Returns an instance of a TechnicalInformation object, containing 
   * detailed information about the technical background of this class,
   * e.g., paper reference or book this class is based on.
   * 
   * @return the technical information about this class
   */
  @Override
  public TechnicalInformation getTechnicalInformation() {
    TechnicalInformation 	result;
    
    result = new TechnicalInformation(Type.ARTICLE);
    result.setValue(Field.AUTHOR, "Leo Breiman");
    result.setValue(Field.YEAR, "1996");
    result.setValue(Field.TITLE, "Bagging predictors");
    result.setValue(Field.JOURNAL, "Machine Learning");
    result.setValue(Field.VOLUME, "24");
    result.setValue(Field.NUMBER, "2");
    result.setValue(Field.PAGES, "123-140");
    
    return result;
  }

  /**
   * String describing default classifier.
   * 
   * @return the default classifier classname
   */
  @Override
  protected String defaultClassifierString() {

    return "weka.classifiers.trees.REPTree";
  }

  /**
   * Returns an enumeration describing the available options.
   *
   * @return an enumeration of all the available options.
   */
  @Override
  public Enumeration<Option> listOptions() {

    Vector<Option> newVector = new Vector<Option>(4);

    newVector.addElement(new Option(
              "\tSize of each bag, as a percentage of the\n" 
              + "\ttraining set size. (default 100)",
              "P", 1, "-P"));
    newVector.addElement(new Option(
              "\tCalculate the out of bag error.",
              "O", 0, "-O"));
    newVector.addElement(new Option(
              "\tWhether to store out of bag predictions in internal evaluation object.",
              "store-out-of-bag-predictions", 0, "-store-out-of-bag-predictions"));
    newVector.addElement(new Option(
              "\tWhether to output complexity-based statistics when out-of-bag evaluation is performed.",
              "output-out-of-bag-complexity-statistics", 0, "-output-out-of-bag-complexity-statistics"));
    newVector.addElement(new Option(
              "\tRepresent copies of instances using weights rather than explicitly.",
              "represent-copies-using-weights", 0, "-represent-copies-using-weights"));
    newVector.addElement(new Option(
              "\tPrint the individual classifiers in the output", "print", 0, "-print"));
    newVector.addElement(new Option(
            "\tClassify using the Majority Voting combination rule",
            "V", 0, "-V"));

    newVector.addAll(Collections.list(super.listOptions()));
 
    return newVector.elements();
  }


  /**
   * Parses a given list of options. <p/>
   *
   <!-- options-start -->
   * Valid options are: <p/>
   * 
   * <pre> -P
   *  Size of each bag, as a percentage of the
   *  training set size. (default 100)</pre>
   * 
   * <pre> -O
   *  Calculate the out of bag error.</pre>
   *
   * <pre> -print
   *  Print the individual classifiers in the output</pre>
   *
   * <pre> -store-out-of-bag-predictions
   *  Whether to store out of bag predictions in internal evaluation object.</pre>
   *
   * <pre> -output-out-of-bag-complexity-statistics
   *  Whether to output complexity-based statistics when out-of-bag evaluation is performed.</pre>
   *
   * <pre> -represent-copies-using-weights
   *  Represent copies of instances using weights rather than explicitly.</pre>
   * 
   * <pre> -S &lt;num&gt;
   *  Random number seed.
   *  (default 1)</pre>
   * 
   * <pre> -num-slots &lt;num&gt;
   *  Number of execution slots.
   *  (default 1 - i.e. no parallelism)</pre>
   * 
   * <pre> -I &lt;num&gt;
   *  Number of iterations.
   *  (default 10)</pre>
   * 
   * <pre> -D
   *  If set, classifier is run in debug mode and
   *  may output additional info to the console</pre>
   * 
   * <pre> -W
   *  Full name of base classifier.
   *  (default: weka.classifiers.trees.REPTree)</pre>
   * 
   * <pre> 
   * Options specific to classifier weka.classifiers.trees.REPTree:
   * </pre>
   * 
   * <pre> -M &lt;minimum number of instances&gt;
   *  Set minimum number of instances per leaf (default 2).</pre>
   * 
   * <pre> -V &lt;minimum variance for split&gt;
   *  Set minimum numeric class variance proportion
   *  of train variance for split (default 1e-3).</pre>
   * 
   * <pre> -N &lt;number of folds&gt;
   *  Number of folds for reduced error pruning (default 3).</pre>
   * 
   * <pre> -S &lt;seed&gt;
   *  Seed for random data shuffling (default 1).</pre>
   * 
   * <pre> -P
   *  No pruning.</pre>
   * 
   * <pre> -L
   *  Maximum tree depth (default -1, no maximum)</pre>
   * 
   * <pre> -I
   *  Initial class value count (default 0)</pre>
   * 
   * <pre> -R
   *  Spread initial count over all class values (i.e. don't use 1 per value)</pre>
   *
   * <pre> -V
   *  Classify using the Majority Voting combination rule.</pre>
   *
   <!-- options-end -->
   *
   * Options after -- are passed to the designated classifier.<p>
   *
   * @param options the list of options as an array of strings
   * @throws Exception if an option is not supported
   */
  @Override
  public void setOptions(String[] options) throws Exception {

    String bagSize = Utils.getOption('P', options);
    if (bagSize.length() != 0) {
      setBagSizePercent(Integer.parseInt(bagSize));
    } else {
      setBagSizePercent(100);
    }

    setCalcOutOfBag(Utils.getFlag('O', options));

    setStoreOutOfBagPredictions(Utils.getFlag("store-out-of-bag-predictions", options));

    setOutputOutOfBagComplexityStatistics(Utils.getFlag("output-out-of-bag-complexity-statistics", options));

    setRepresentCopiesUsingWeights(Utils.getFlag("represent-copies-using-weights", options));

    setPrintClassifiers(Utils.getFlag("print", options));
    
    setMajorityVotingRule(Utils.getFlag('V', options));

    super.setOptions(options);
  }

  /**
   * Gets the current settings of the Classifier.
   *
   * @return an array of strings suitable for passing to setOptions
   */
  @Override
  public String [] getOptions() {

    Vector<String> options = new Vector<String>();
    
    options.add("-P"); 
    options.add("" + getBagSizePercent());

    if (getCalcOutOfBag()) { 
        options.add("-O");
    }

    if (getStoreOutOfBagPredictions()) {
        options.add("-store-out-of-bag-predictions");
    }

    if (getOutputOutOfBagComplexityStatistics()) {
        options.add("-output-out-of-bag-complexity-statistics");
    }

    if (getRepresentCopiesUsingWeights()) {
        options.add("-represent-copies-using-weights");
    }

    if (getPrintClassifiers()) {
      options.add("-print");
    }

    if (getMajorityVotingRule()) { 
        options.add("-V");
    }

    Collections.addAll(options, super.getOptions());
    
    return options.toArray(new String[0]);
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String bagSizePercentTipText() {
    return "Size of each bag, as a percentage of the training set size.";
  }

  /**
   * Gets the size of each bag, as a percentage of the training set size.
   *
   * @return the bag size, as a percentage.
   */
  public int getBagSizePercent() {

    return m_BagSizePercent;
  }
  
  /**
   * Sets the size of each bag, as a percentage of the training set size.
   *
   * @param newBagSizePercent the bag size, as a percentage.
   */
  public void setBagSizePercent(int newBagSizePercent) {

    m_BagSizePercent = newBagSizePercent;
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String representCopiesUsingWeightsTipText() {
    return "Whether to represent copies of instances using weights rather than explicitly.";
  }

  /**
   * Set whether copies of instances are represented using weights rather than explicitly.
   *
   * @param representUsingWeights whether to represent copies using weights
   */
  public void setRepresentCopiesUsingWeights(boolean representUsingWeights) {

    m_RepresentUsingWeights = representUsingWeights;
  }

  /**
   * Get whether copies of instances are represented using weights rather than explicitly.
   *
   * @return whether copies of instances are represented using weights rather than explicitly
   */
  public boolean getRepresentCopiesUsingWeights() {

    return m_RepresentUsingWeights;
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String storeOutOfBagPredictionsTipText() {
    return "Whether to store the out-of-bag predictions.";
  }

  /**
   * Set whether the out of bag predictions are stored.
   *
   * @param storeOutOfBag whether the out of bag predictions are stored
   */
  public void setStoreOutOfBagPredictions(boolean storeOutOfBag) {

    m_StoreOutOfBagPredictions = storeOutOfBag;
  }

  /**
   * Get whether the out of bag predictions are stored.
   *
   * @return whether the out of bag predictions are stored
   */
  public boolean getStoreOutOfBagPredictions() {

    return m_StoreOutOfBagPredictions;
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String calcOutOfBagTipText() {
    return "Whether the out-of-bag error is calculated.";
  }

  /**
   * Set whether the out of bag error is calculated.
   *
   * @param calcOutOfBag whether to calculate the out of bag error
   */
  public void setCalcOutOfBag(boolean calcOutOfBag) {

    m_CalcOutOfBag = calcOutOfBag;
  }

  /**
   * Get whether the out of bag error is calculated.
   *
   * @return whether the out of bag error is calculated
   */
  public boolean getCalcOutOfBag() {

    return m_CalcOutOfBag;
  }
  
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String majorityVotingRuleTipText() {
    return "Whether to classify using the Majority Voting combination rule is used.";
  }

  /**
   * Set whether to classify using the Majority Voting combination rule. 
   *
   * @param majorityVotingRule whether to classify using the Majority Voting combination rule
   */
  public void setMajorityVotingRule(boolean majorityVotingRule) {

    m_MajorityVotingRule = majorityVotingRule;
  }

  /**
   * Get whether to classify using the Majority Voting combination rule is used.
   *
   * @return whether to classify using the Majority Voting combination rule is used
   */
  public boolean getMajorityVotingRule() {

    return m_MajorityVotingRule;
  }
  
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String outputOutOfBagComplexityStatisticsTipText() {

    return "Whether to output complexity-based statistics when out-of-bag evaluation is performed.";
  }

  /**
   * Gets whether complexity statistics are output when OOB estimation is performed.
   *
   * @return whether statistics are calculated
   */
  public boolean getOutputOutOfBagComplexityStatistics() {

    return m_OutputOutOfBagComplexityStatistics;
  }

  /**
   * Sets whether complexity statistics are output when OOB estimation is performed.
   *
   * @param b whether statistics are calculated
   */
  public void setOutputOutOfBagComplexityStatistics(boolean b) {

    m_OutputOutOfBagComplexityStatistics = b;
  }

  /**
   * Returns the tip text for this property
   *
   * @return tip text for this property suitable for displaying in the
   *         explorer/experimenter gui
   */
  public String printClassifiersTipText() {
    return "Print the individual classifiers in the output";
  }

  /**
   * Set whether to print the individual ensemble classifiers in the output
   *
   * @param print true if the individual classifiers are to be printed
   */
  public void setPrintClassifiers(boolean print) {
    m_printClassifiers = print;
  }

  /**
   * Get whether to print the individual ensemble classifiers in the output
   *
   * @return true if the individual classifiers are to be printed
   */
  public boolean getPrintClassifiers() {
    return m_printClassifiers;
  }

  /**
   * Gets the out of bag error that was calculated as the classifier
   * was built. Returns error rate in classification case and
   * mean absolute error in regression case.
   *
   * @return the out of bag error; -1 if out-of-bag-error has not be estimated
   */
  public double measureOutOfBagError() {

    if (m_OutOfBagEvaluationObject == null) {
      return -1;
    }
    if (m_Numeric) {
      return m_OutOfBagEvaluationObject.meanAbsoluteError();
    } else {
      return m_OutOfBagEvaluationObject.errorRate();
    }
  }
  
  /**
   * Returns an enumeration of the additional measure names
   * (Added also those produced by the base algorithm).
   * 
   * @return an enumeration of the measure names
   */
  @Override
  public Enumeration<String> enumerateMeasures() {
    
    Vector<String> newVector = new Vector<String>(1);
    newVector.addElement("measureOutOfBagError");
    if (m_Classifier instanceof J48) {
    	String[] stMetaOperations = new String[]{"Avg", "Min", "Max", "Sum", "Mdn", "Dev"};
    	ArrayList<String> metaOperations = new ArrayList<>(Arrays.asList(stMetaOperations));
    	String[] stTreeMeasures = new String[]{"NumLeaves", "NumRules", "NumInnerNodes", "ExplanationLength", "WeightedExplanationLength"};
    	ArrayList<String> treeMeasures = new ArrayList<>(Arrays.asList(stTreeMeasures));

    	for(String op: metaOperations)
    		for(String ms: treeMeasures)
    			newVector.addElement("measure" + op + ms);
    }
    return newVector.elements();
  }
  
	/**
	 * 	Returns the value of a named aggregate measure associated to
	 *  a meta-classifier based on a set of decision trees.
	 *
	 * @param singleMeasureName the name of the measure to query for its value
	 * @param iop identifies the type of aggregated operation
	 * @return the value of the named measure
	 * @throws IllegalArgumentException if the named measure is not supported
	 */
	public double getMetaAggregateMeasure(String singleMeasureName, int iop) {
		double res = (double)0.0;
		double[] vValues = new double[m_Classifiers.length];
		for(int i = 0; i < m_Classifiers.length; i++) {
			vValues[i] = ((AdditionalMeasureProducer)(m_Classifiers[i])).getMeasure(singleMeasureName);
		}
		switch (iop) {
			case 0: // Avg
				res = Utils.mean(vValues); break;
			case 1: // Min
				res = vValues[Utils.minIndex(vValues)]; break;
			case 2: // Max
				res = vValues[Utils.maxIndex(vValues)]; break;
			case 3: // Sum
				res = Utils.sum(vValues); break;
			case 4: // Mdn
				// // TODO median could be a method to insert in the class 'DoubleVector'
				DoubleVector auxValues = new DoubleVector(vValues);
				auxValues.sort();
				res = auxValues.get(((m_Classifiers.length+1)/2)-1);
				break;
			case 5: // Dev 
				res = Math.sqrt(Utils.variance(vValues)); break;
		}
		return res;
	}

  /**
   * Returns the value of the named measure.
   * In the case of measureTreeSize, measureNumLeaves, measureNumRules,
   * measureExplanationLength and measureWeightedExplanationLength it returns
   * the sum of the named measure of all the base classifiers. 
   *
   * @param additionalMeasureName the name of the measure to query for its value
   * @return the value of the named measure
   * @throws IllegalArgumentException if the named measure is not supported
   */
  @Override
  public double getMeasure(String additionalMeasureName) {

	  if (additionalMeasureName.equalsIgnoreCase("measureOutOfBagError")) {
		  return measureOutOfBagError();
	  } else if (m_Classifier instanceof J48) {
		  String[] stMetaOperations = new String[]{"Avg", "Min", "Max", "Sum", "Mdn", "Dev"};
		  ArrayList<String> metaOperations = new ArrayList<>(Arrays.asList(stMetaOperations));
		  String[] stTreeMeasures = new String[]{"NumLeaves", "NumRules", "NumInnerNodes", "ExplanationLength", "WeightedExplanationLength"};
		  ArrayList<String> treeMeasures = new ArrayList<>(Arrays.asList(stTreeMeasures));

		  for(String op: metaOperations)
			  for(String ms: treeMeasures) {
				  String mtms = "measure" + op + ms;
				  if (additionalMeasureName.equalsIgnoreCase(mtms)) {
					  String sms = "measure" + ms;
					  int iop = metaOperations.indexOf(op);
					  return getMetaAggregateMeasure(sms, iop);
				  }
			  }
		  // Measures associated to a single decision tree are not defined
		  return Double.NaN;
	  }
	  else throw new IllegalArgumentException(additionalMeasureName 
			  + " not supported (Bagging)");
  }

  /**
   * Returns a training set for a particular iteration.
   * 
   * @param iteration the number of the iteration for the requested training set.
   * @return the training set for the supplied iteration number
   * @throws Exception if something goes wrong when generating a training set.
   */
  @Override
  protected synchronized Instances getTrainingSet(int iteration) throws Exception {

    Random r = new Random(m_Seed + iteration);

    // create the in-bag indicator array if necessary
    if (m_CalcOutOfBag) {
      m_inBag[iteration] = new boolean[m_data.numInstances()];
      return m_data.resampleWithWeights(r, m_inBag[iteration], getRepresentCopiesUsingWeights(), m_BagSizePercent);
    } else {
      return m_data.resampleWithWeights(r, null, getRepresentCopiesUsingWeights(), m_BagSizePercent);
    }
  }

  /**
   * Returns the out-of-bag evaluation object.
   *
   * @return the out-of-bag evaluation object; null if out-of-bag error hasn't been calculated
   */
  public Evaluation getOutOfBagEvaluationObject() {

    return m_OutOfBagEvaluationObject;
  }

  /**
   * Bagging method.
   *
   * @param data the training data to be used for generating the
   * bagged classifier.
   * @throws Exception if the classifier could not be built successfully
   */
  @Override
  public void buildClassifier(Instances data) throws Exception {

	long baggingTimeStart, baggingTimeElapsed;
    // can classifier handle the data?
    getCapabilities().testWithFail(data);

    // Has user asked to represent copies using weights?
    if (getRepresentCopiesUsingWeights() && !(m_Classifier instanceof WeightedInstancesHandler)) {
      throw new IllegalArgumentException("Cannot represent copies using weights when " +
              "base learner in bagging does not implement " +
              "WeightedInstancesHandler.");
    }

    // get fresh Instances object
    m_data = new Instances(data);

    super.buildClassifier(m_data);

    m_random = new Random(m_Seed);

    m_inBag = null;
    if (m_CalcOutOfBag)
      m_inBag = new boolean[m_Classifiers.length][];

    for (int j = 0; j < m_Classifiers.length; j++) {
      if (m_Classifier instanceof Randomizable) {
        ((Randomizable) m_Classifiers[j]).setSeed(m_random.nextInt());
      }
    }

    m_Numeric = m_data.classAttribute().isNumeric();
    
    baggingTimeStart = System.currentTimeMillis();
    
    buildClassifiers();
    
    baggingTimeElapsed = System.currentTimeMillis() - baggingTimeStart;
    
    System.out.println("Bagging denbora: " + baggingTimeElapsed + " ms \n");

    // calc OOB error?
    if (getCalcOutOfBag()) {
      m_OutOfBagEvaluationObject = new Evaluation(m_data);

      for (int i = 0; i < m_data.numInstances(); i++) {
        double[] votes;
        if (m_Numeric)
          votes = new double[1];
        else
          votes = new double[m_data.numClasses()];

        // determine predictions for instance
        int voteCount = 0;
        for (int j = 0; j < m_Classifiers.length; j++) {
          if (m_inBag[j][i])
            continue;

          if (m_Numeric) {
            double pred = m_Classifiers[j].classifyInstance(m_data.instance(i));
            if (!Utils.isMissingValue(pred)) {
              votes[0] += pred;
              voteCount++;
            }
          } else {
            voteCount++;
            double[] newProbs = m_Classifiers[j].distributionForInstance(m_data.instance(i));
            // sum the probability estimates
            for (int k = 0; k < newProbs.length; k++) {
              votes[k] += newProbs[k];
            }
          }
        }

        // "vote"
        if (m_Numeric) {
          if (voteCount > 0) {
            votes[0] /= voteCount;
            m_OutOfBagEvaluationObject.evaluationForSingleInstance(votes, m_data.instance(i), getStoreOutOfBagPredictions());
          }
        } else {
          double sum = Utils.sum(votes);
          if (sum > 0) {
            Utils.normalize(votes, sum);
            m_OutOfBagEvaluationObject.evaluationForSingleInstance(votes, m_data.instance(i), getStoreOutOfBagPredictions());
          }
        }
      }
    } else {
      m_OutOfBagEvaluationObject = null;
    }

    // save memory
    m_inBag = null;
    m_data = new Instances(m_data, 0);
  }

  /**
   * Calculates the class membership probabilities for the given test
   * instance.
   *
   * @param instance the instance to be classified
   * @return preedicted class probability distribution
   * @throws Exception if distribution can't be computed successfully 
   */
  @Override
  public double[] distributionForInstance(Instance instance) throws Exception {

    double[] sums = new double[instance.numClasses()], newProbs;

    double numPreds = 0;
    for (int i = 0; i < m_NumIterations; i++) {
      if (m_Numeric) {
        double pred = m_Classifiers[i].classifyInstance(instance);
        if (!Utils.isMissingValue(pred)) {
          sums[0] += pred;
          numPreds++;
        }
      } else {
    	  if (m_MajorityVotingRule) {
    		  double pred = m_Classifiers[i].classifyInstance(instance);
    		  if (!Utils.isMissingValue(pred)) {
    			  sums[(int)pred]++;
    			  numPreds++;
    		  }
    	  } else {
    		  newProbs = m_Classifiers[i].distributionForInstance(instance);
    		  for (int j = 0; j < newProbs.length; j++)
    			  sums[j] += newProbs[j];
    	  }
      }
    }
    if (m_Numeric) {
      if (numPreds == 0) {
        sums[0] = Utils.missingValue();
      } else {
        sums[0] /= numPreds;
      }
      return sums;
    } else if (m_MajorityVotingRule) {
		int mostVotedClass = Utils.maxIndex(sums);
	    // set probs to 0
		newProbs = new double[instance.numClasses()];

		newProbs[mostVotedClass] = 1; // the class that have been voted the most
	                              // receives 1
	    return newProbs;
    } else {
        if (Utils.eq(Utils.sum(sums), 0)) {
            return sums;
          } else {
            Utils.normalize(sums);
            return sums;
          }
    }
  }

  /**
   * Tool tip text for this property
   *
   * @return the tool tip for this property
   */
  public String batchSizeTipText() {
    return "Batch size to use if base learner is a BatchPredictor";
  }

  /**
   * Set the batch size to use. Gets passed through to the base learner if it
   * implements BatchPredictor. Otherwise it is just ignored.
   *
   * @param size the batch size to use
   */
  public void setBatchSize(String size) {

    if (getClassifier() instanceof BatchPredictor) {
      ((BatchPredictor) getClassifier()).setBatchSize(size);
    } else {
      super.setBatchSize(size);
    }
  }

  /**
   * Gets the preferred batch size from the base learner if it implements
   * BatchPredictor. Returns 1 as the preferred batch size otherwise.
   *
   * @return the batch size to use
   */
  public String getBatchSize() {

    if (getClassifier() instanceof BatchPredictor) {
      return ((BatchPredictor) getClassifier()).getBatchSize();
    } else {
      return super.getBatchSize();
    }
  }

  /**
   * Batch scoring method. Calls the appropriate method for the base learner if
   * it implements BatchPredictor. Otherwise it simply calls the
   * distributionForInstance() method repeatedly.
   *
   * @param insts the instances to get predictions for
   * @return an array of probability distributions, one for each instance
   * @throws Exception if a problem occurs
   */
  public double[][] distributionsForInstances(Instances insts)
          throws Exception {


    if (getClassifier() instanceof BatchPredictor) {

      ExecutorService pool = Executors.newFixedThreadPool(m_numExecutionSlots);

      // Set up result set, and chunk size
      final int chunksize = m_Classifiers.length / m_numExecutionSlots;
      Set<Future<double[][]>> results = new HashSet<Future<double[][]>>();

      // For each thread
      for (int j = 0; j < m_numExecutionSlots; j++) {

        // Determine batch to be processed
        final int lo = j * chunksize;
        final int hi = (j < m_numExecutionSlots - 1) ? (lo + chunksize) : m_Classifiers.length;

        // Create and submit new job for each batch of instances
        Future<double[][]> futureT = pool.submit(new Callable<double[][]>() {
          @Override
          public double[][] call() throws Exception {
            if (insts.classAttribute().isNumeric()) {
              double[][] ensemblePreds = new double[insts.numInstances()][2];
              for (int i = lo; i < hi; i++) {
                double[][] preds = ((BatchPredictor) m_Classifiers[i]).distributionsForInstances(insts);
                for (int j = 0; j < preds.length; j++) {
                  if (!Utils.isMissingValue(preds[j][0])) {
                    ensemblePreds[j][0] += preds[j][0];
                    ensemblePreds[j][1]++;
                  }
                }
              }
              return ensemblePreds;
            } else {
              double[][] ensemblePreds = new double[insts.numInstances()][insts.numClasses()];
              for (int i = lo; i < hi; i++) {
                double[][] preds = ((BatchPredictor) m_Classifiers[i]).distributionsForInstances(insts);
                for (int j = 0; j < preds.length; j++) {
                  for (int k = 0; k < preds[j].length; k++) {
                    ensemblePreds[j][k] += preds[j][k];
                  }
                }
              }
              return ensemblePreds;
            }
          }
        });
        results.add(futureT);
      }

      // Form ensemble prediction
      double[][] ensemblePreds =
              new double[insts.numInstances()][insts.classAttribute().isNumeric() ? 2 : insts.numClasses()];
      try {
        for (Future<double[][]> futureT : results) {
          double[][] preds = futureT.get();
          for (int j = 0; j < preds.length; j++) {
            for (int k = 0; k < preds[j].length; k++) {
              ensemblePreds[j][k] += preds[j][k];
            }
          }
        }
      } catch (Exception e) {
        System.out.println("RandomCommittee: predictions could not be generated by thread.");
        e.printStackTrace();
      }
      pool.shutdown();

      // Normalise ensemble predictions
      if (insts.classAttribute().isNumeric() == true) {
        double[][] finalPreds = new double[ensemblePreds.length][1];
        for (int j = 0; j < ensemblePreds.length; j++) {
          if (ensemblePreds[j][1] == 0) {
            finalPreds[j][0] = Utils.missingValue();
          } else {
            finalPreds[j][0] = ensemblePreds[j][0] / ensemblePreds[j][1];
          }
        }
        return finalPreds;
      } else {
        for (int j = 0; j < ensemblePreds.length; j++) {
          double sum = Utils.sum(ensemblePreds[j]);
          if (!Utils.eq((sum), 0)) {
            Utils.normalize(ensemblePreds[j], sum);
          }
        }
        return ensemblePreds;
      }
    } else {

      /** Multi-threading in this branch causes issues

       // Set up result set, and chunk size
       final int chunksize = insts.numInstances() / m_numExecutionSlots;
       Set<Future<Void>> results = new HashSet<Future<Void>>();

       final double[][] ensemblePreds = new double[insts.numInstances()][insts.numClasses()];

       // For each thread
       for (int j = 0; j < m_numExecutionSlots; j++) {

       // Determine batch to be processed
       final int lo = j * chunksize;
       final int hi = (j < m_numExecutionSlots - 1) ? (lo + chunksize) : insts.numInstances();

       // Create and submit new job for each batch of instances
       Future<Void> futureT = pool.submit(new Callable<Void>() {
      @Override
      public Void call() throws Exception {
      for (int i = lo; i < hi; i++) {
      System.arraycopy(distributionForInstance((Instance)insts.instance(i).copy()), 0,
      ensemblePreds[i], 0, insts.numClasses());
      }
      return null;
      }
      });
       results.add(futureT);
       }

       // Incorporate predictions
       try {
       for (Future<Void> futureT : results) {
       futureT.get();
       }
       } catch (Exception e) {
       System.out.println("RandomCommittee: predictions could not be generated by thread.");
       e.printStackTrace();
       }
       pool.shutdown();

       return ensemblePreds;*/
      double[][] result = new double[insts.numInstances()][insts.numClasses()];
      for (int i = 0; i < insts.numInstances(); i++) {
        result[i] = distributionForInstance(insts.instance(i));
      }
      return result;
    }
  }

  /**
   * Returns true if the base classifier implements BatchPredictor and is able
   * to generate batch predictions efficiently
   *
   * @return true if the base classifier can generate batch predictions efficiently
   */
  public boolean implementsMoreEfficientBatchPrediction() {
    if (!(getClassifier() instanceof BatchPredictor)) {
      return super.implementsMoreEfficientBatchPrediction();
    }
    return ((BatchPredictor) getClassifier()).implementsMoreEfficientBatchPrediction();
  }

  /**
   * Returns description of the bagged classifier.
   *
   * @return description of the bagged classifier as a string
   */
  @Override
  public String toString() {
    
    if (m_Classifiers == null) {
      return "Bagging: No model built yet.";
    }
    StringBuffer text = new StringBuffer();
    text.append("Bagging with " + getNumIterations() + " iterations and base learner\n\n" + getClassifierSpec());
    if (getPrintClassifiers()) {
      text.append("All the base classifiers: \n\n");
      for (int i = 0; i < m_Classifiers.length; i++)
        text.append(m_Classifiers[i].toString() + "\n");

      text.append("\n--- Complexity/Explanation measures  ---");
      text.append("\n--- of the whole multiple classifier ---");
      text.append("\n----------------------------------------\n");
			
	  String[] stMetaOperations = new String[]{"Avg", "Min", "Max", "Sum", "Mdn", "Dev"};
	  ArrayList<String> metaOperations = new ArrayList<>(Arrays.asList(stMetaOperations));
	  String[] stTreeMeasures = new String[]{"NumLeaves", /*"NumRules",*/ "NumInnerNodes", "ExplanationLength", "WeightedExplanationLength"};
	  ArrayList<String> treeMeasures = new ArrayList<>(Arrays.asList(stTreeMeasures));
	  
	  for(String ms: treeMeasures) {
	      text.append(ms + ": ");
		  for(String op: metaOperations) {
			  String mtms = "measure" + op + ms;
		      text.append(op + " = " + Utils.roundDouble(getMeasure(mtms),2) + " ");
		  }
	      text.append("\n");
	  }
      text.append("\n");
    }
    if (m_CalcOutOfBag) {
      text.append(m_OutOfBagEvaluationObject.toSummaryString("\n\n*** Out-of-bag estimates ***\n", getOutputOutOfBagComplexityStatistics()));
    }

    return text.toString();
  }
  
  /**
   * Builds the classifier to generate a partition.
   */
  @Override
  public void generatePartition(Instances data) throws Exception {
    
    if (m_Classifier instanceof PartitionGenerator)
      buildClassifier(data);
    else throw new Exception("Classifier: " + getClassifierSpec()
			     + " cannot generate a partition");
  }
  
  /**
   * Computes an array that indicates leaf membership
   */
  @Override
  public double[] getMembershipValues(Instance inst) throws Exception {
    
    if (m_Classifier instanceof PartitionGenerator) {
      ArrayList<double[]> al = new ArrayList<double[]>();
      int size = 0;
      for (int i = 0; i < m_Classifiers.length; i++) {
        double[] r = ((PartitionGenerator)m_Classifiers[i]).
          getMembershipValues(inst);
        size += r.length;
        al.add(r);
      }
      double[] values = new double[size];
      int pos = 0;
      for (double[] v: al) {
        System.arraycopy(v, 0, values, pos, v.length);
        pos += v.length;
      }
      return values;
    } else throw new Exception("Classifier: " + getClassifierSpec()
                               + " cannot generate a partition");
  }
  
  /**
   * Returns the number of elements in the partition.
   */
  @Override
  public int numElements() throws Exception {
    
    if (m_Classifier instanceof PartitionGenerator) {
      int size = 0;
      for (int i = 0; i < m_Classifiers.length; i++) {
        size += ((PartitionGenerator)m_Classifiers[i]).numElements();
      }
      return size;
    } else throw new Exception("Classifier: " + getClassifierSpec()
                               + " cannot generate a partition");
  }
  
  /**
   * Returns the revision string.
   * 
   * @return		the revision
   */
  @Override
  public String getRevision() {
    return RevisionUtils.extract("$Revision: 15801 $");
  }

  /**
   * Main method for testing this class.
   *
   * @param argv the options
   */
  public static void main(String [] argv) {
    runClassifier(new Bagging(), argv);
  }
  
  protected List<Classifier> m_classifiersCache;

  /**
   * Aggregate an object with this one
   * 
   * @param toAggregate the object to aggregate
   * @return the result of aggregation
   * @throws Exception if the supplied object can't be aggregated for some
   *           reason
   */
  @Override
  public Bagging aggregate(Bagging toAggregate) throws Exception {
    if (!m_Classifier.getClass().isAssignableFrom(toAggregate.m_Classifier.getClass())) {
      throw new Exception("Can't aggregate because base classifiers differ");
    }
    
    if (m_classifiersCache == null) {
      m_classifiersCache = new ArrayList<Classifier>();
      m_classifiersCache.addAll(Arrays.asList(m_Classifiers));
    }
    m_classifiersCache.addAll(Arrays.asList(toAggregate.m_Classifiers));
    
    return this;
  }

  /**
   * Call to complete the aggregation process. Allows implementers to do any
   * final processing based on how many objects were aggregated.
   * 
   * @throws Exception if the aggregation can't be finalized for some reason
   */
  @Override
  public void finalizeAggregation() throws Exception {    
    m_Classifiers = m_classifiersCache.toArray(new Classifier[1]);
    m_NumIterations = m_Classifiers.length;
    
    m_classifiersCache = null;
  }
}

