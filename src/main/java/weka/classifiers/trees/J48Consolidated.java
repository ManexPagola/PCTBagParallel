/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    J48Consolidated.java
 *    Copyright (C) 2021 ALDAPA Team (http://www.aldapa.eus)
 *    Faculty of Informatics, Donostia, 20018
 *    University of the Basque Country (UPV/EHU), Basque Country
 *    
 */

package weka.classifiers.trees;

import java.util.Collections;
import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;

import weka.classifiers.Sourcable;
import weka.classifiers.trees.j48.C45ModelSelection;
import weka.classifiers.trees.j48.ModelSelection;
import weka.classifiers.trees.j48Consolidated.C45ConsolidatedModelSelection;
import weka.classifiers.trees.j48Consolidated.C45ConsolidatedPruneableClassifierTree;
import weka.classifiers.trees.j48Consolidated.InstancesConsolidated;
import weka.core.AdditionalMeasureProducer;
import weka.core.Drawable;
import weka.core.Instances;
import weka.core.Matchable;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.SelectedTag;
import weka.core.Summarizable;
import weka.core.Tag;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.gui.ProgrammaticProperty;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

/**
<!-- globalinfo-start -->
 * Class for generating a pruned or unpruned C4.5 consolidated tree. Uses the Consolidated Tree Construction (CTC) algorithm: a single tree is built based on a set of subsamples. New options are added to the J48 class to set the Resampling Method (RM) for the generation of samples to be used in the consolidation process. For more information, see:<br/>
 * <br/>
 * Jes&uacute;s M. P&eacute;rez and Javier Muguerza and Olatz Arbelaitz and Ibai Gurrutxaga and Jos&eacute; I. Mart&iacute;­n.  
 * "Combining multiple class distribution modified subsamples in a single tree". Pattern Recognition Letters (2007), 28(4), pp 414-422.
 * <a href="http://dx.doi.org/10.1016/j.patrec.2006.08.013" target="_blank">doi:10.1016/j.patrec.2006.08.013</a>
 * <p/>
 * A new way has been added to determine the number of samples to be used in the consolidation process which guarantees the minimum percentage, the coverage value, of the examples of the original sample to be contained by the set of built subsamples. For more information, see:<br/>
 * <br/>
 * Igor Ibarguren and Jes&uacute;s M. P&eacute;rez and Javier Muguerza and Ibai Gurrutxaga and Olatz Arbelaitz.  
 * "Coverage-based resampling: Building robust consolidated decision trees". Knowledge Based Systems (2015), Vol. 79, pp 51-67.
 * <a href="http://dx.doi.org/10.1016/j.knosys.2014.12.023" target="_blank">doi:10.1016/j.knosys.2014.12.023</a>
 * <p/>
<!-- globalinfo-end -->
 *
<!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;article{Perez2007,
 *    title = "Combining multiple class distribution modified subsamples in a single tree",
 *    journal = "Pattern Recognition Letters",
 *    volume = "28",
 *    number = "4",
 *    pages = "414 - 422",
 *    year = "2007",
 *    doi = "10.1016/j.patrec.2006.08.013",
 *    author = "Jes\'us M. P\'erez and Javier Muguerza and Olatz Arbelaitz and Ibai Gurrutxaga and Jos\'e I. Mart\'i­n"
 * }
 * </pre>
 * <p/>
 * <pre>
 * &#64;article{Ibarguren2015,
 *    title = "Coverage-based resampling: Building robust consolidated decision trees",
 *    journal = "Knowledge Based Systems",
 *    volume = "79",
 *    pages = "51 - 67",
 *    year = "2015",
 *    doi = "10.1016/j.knosys.2014.12.023",
 *    author = "Igor Ibarguren and Jes\'us M. P\'erez and Javier Muguerza and Ibai Gurrutxaga and Olatz Arbelaitz"
 * }
 * </pre>
 * <p/>
<!-- technical-bibtex-end -->
 * *************************************************************************************<br/>
<!-- options-start -->
 * Valid options are: <p/>
 * 
 * J48 options <br/>
 * ==========
 *
 * <pre>
 * -U
 *  Use unpruned tree.
 * </pre>
 * 
 * <pre>
 * -O
 *  Do not collapse tree.
 * </pre>
 * 
 * <pre>
 * -C &lt;pruning confidence&gt;
 *  Set confidence threshold for pruning.
 *  (default 0.25)
 * </pre>
 * 
 * <pre>
 * -M &lt;minimum number of instances&gt;
 *  Set minimum number of instances per leaf.
 *  (default 2)
 *  </pre>
 *  
 * <pre>
 * -S
 *  Don't perform subtree raising.
 * </pre>
 * 
 * <pre>
 * -L
 *  Do not clean up after the tree has been built.
 * </pre>
 * 
 * <pre>
 * -J
 *  Do not use MDL correction for info gain on numeric attributes.
 * </pre>
 * 
 * <pre>
 * -A
 *  Laplace smoothing for predicted probabilities.
 * </pre>
 * 
 * <pre>
 * -Q &lt;seed&gt;
 *  Seed for random data shuffling (default 1).
 * </pre>
 * 
 * <pre>
 * -doNotMakeSplitPointActualValue
 *  Do not make split point actual value.
 * </pre>
 * 
 * Options to set the Resampling Method (RM) for the generation of samples
 *  to use in the consolidation process <br/>
 * =============================================================================================== 
 * <pre> -RM-C
 *  Determines the way to set the number of samples to be generated will be based on
 *  a coverage value as a percentage. In the case this option is not set, the number of samples
 *  will be determined using a fixed value. 
 *  (set by default)</pre>
 * 
 * <pre> -RM-N &lt;number of samples&gt;
 *  Number of samples to be generated for the use in the construction of the consolidated tree.
 *  It can be set as a fixed value or based on a coverage value as a percentage, when -RM-C option
 *  is used, which guarantees the number of samples necessary to adequately cover the examples 
 *  of the original sample
 *  (default 5 for a fixed value or 99% for the case based on a coverage value)</pre>
 * 
 * <pre> -RM-R
 *  Determines whether or not replacement is used when generating the samples.
 *  (default false)</pre>
 * 
 * <pre> -RM-B &lt;Size of each sample(&#37;)&gt;
 *  Size of each sample(bag), as a percentage of the training set size.
 *  Combined with the option &lt;distribution minority class&gt; accepts:
 *  * -1 (sizeOfMinClass): The size of the minority class  
 *  * -2 (Max): Maximum size taking &lt;distribution minority class&gt; into account
 *  *           and using no replacement
 *  (default -2)</pre>
 *  
 * <pre> -RM-D &lt;distribution minority class&gt;
 *  Determines the new value of the distribution of the minority class, if we want to change it.
 *  It can be one of the following values:
 *  * A value between 0 and 100 to change the portion of minority class instances in the new samples
 *    (this option can only be used with binary problems (two-class datasets))
 *  * -1 (free): Works with the instances without taking their class into account
 *  * -2 (stratified): Maintains the original class distribution in the new samples
 *  (default 50.0) 
 * 
<!-- options-end -->
 *
 * @author Jes&uacute;s M. P&eacute;rez (txus.perez@ehu.eus)
 * @author Igor Ibarguren (igor.ibarguren@ehu.eus) 
 *  (based on the previous version written in colaboration with Fernando Lozano)
 *  (based on J48.java written by Eibe Frank)
 * @version $Revision: 3.2 $
 */
public class J48Consolidated
extends J48
implements OptionHandler, Drawable, Matchable, Sourcable, 
WeightedInstancesHandler, Summarizable, AdditionalMeasureProducer, 
TechnicalInformationHandler {

	/** for serialization */
	private static final long serialVersionUID = -2647522302468491144L;

	/** The default value set for the percentage of coverage estimated necessary to adequately cover
	 *  the examples of the original sample with the set of samples to be used in the consolidation process */
	protected static float m_coveragePercent = (float)99;

	/** Number of samples necessary based on coverage (if this option is used) */
	int m_numberSamplesByCoverage = 0;

	/** The true value estimated for the coverage achieved with the set of samples generated
	 *  for the construction of the consolidated tree */
	protected double m_trueCoverage;

	/** Size of each sample(bag), as a percentage of the training set size, to be used in exceptional situations
	 *  where the original class distribution and the distribution of the samples to be generated are the same
	 *  and the size of samples has been set as the maximum possible (maxSize).
	 *  In these cases, all generated samples would have the same examples that the original sample has.
	 *  Due of this the size of samples will be reduced with this value. */
	protected static int m_bagSizePercentToReduce = 75;

	/** Minimum percentage of cases required in each class for the samples to be generated when 
	 *  the distribution of the minority class is changed */
	protected static float m_minExamplesPerClassPercent = (float)2.0;

	/** String containing a brief explanation of exceptional situations, if occur */
	String m_stExceptionalSituationsMessage = "";

	/** Ways to set the numberSamples option */
	public static final int NumberSamples_FixedValue = 1;
	public static final int NumberSamples_BasedOnCoverage = 2;

	/** Strings related to the ways to set the numberSamples option */
	public static final Tag[] TAGS_WAYS_TO_SET_NUMBER_SAMPLES = {
			new Tag(NumberSamples_FixedValue, "using a fixed value"),
			new Tag(NumberSamples_BasedOnCoverage, "based on a coverage value (%)"),
	};

	/** Options to set the Resampling Method (RM) for the generation of samples
	 *   to use in the consolidation process
	 *   (Prefix RM added to the option names in order to appear together in the graphical interface)
	 ********************************************************************************/
	/** Selected way to set the number of samples to be generated; or using a fixed value;
	 *   or based on a coverage value as a percentage (by default). */
	protected int m_RMnumberSamplesHowToSet = NumberSamples_BasedOnCoverage;

	/** Number of samples to be generated for the use in the construction of the consolidated tree.
	 * If m_RMnumberSamplesHowToSet = NumberSamples_BasedOnCoverage, the value of number of 
	 * samples to be used is calculated based on a coverage value in percentage (%), which guarantees
	 * the number of samples necessary to adequately cover the examples of the original sample. */
	protected float m_RMnumberSamples = (float)m_coveragePercent; // default: f(99% of coverage) 

	/** Determines whether or not replacement is used when generating the samples.**/
	protected boolean m_RMreplacement = false;

	/** Size of each sample(bag), as a percentage of the training set size.
	 *  Combined with the option &lt;distribution minority class&gt; accepts:
	 *  * -1 (sizeOfMinClass): The size of the minority class  
	 *  * -2 (maxSize): Maximum size taking &lt;distribution minority class&gt; into account
	 *  *           and using no replacement */
	protected int m_RMbagSizePercent = -2; // default: maxSize

	/** Value of the distribution of the minority class to be changed.
	 * It can be one of the following values: <br>
	 *  * A value between 0 and 100 to change the portion of minority class instances in the new samples
	 *    (If the dataset is multi-class, only the special value 50.0 will be accepted to balance the classes)
	 *  * -1 (free): Works with the instances without taking their class into account
	 *  * -2 (stratified): Maintains the original class distribution in the new samples */
	protected float m_RMnewDistrMinClass = (float)50.0;

	/**
	 * Returns a string describing the classifier
	 * @return a description suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String globalInfo() {
		return "Class for generating a pruned or unpruned C45 consolidated tree. Uses the Consolidated "
				+ "Tree Construction (CTC) algorithm: a single tree is built based on a set of subsamples. "
				+ "New options are added to the J48 class to set the Resampling Method (RM) for "
				+ "the generation of samples to be used in the consolidation process.\n"
				+ "Recently, a new way has been added to determine the number of samples to be used "
				+ "in the consolidation process which guarantees the minimum percentage, the coverage value, "
				+ "of the examples of the original sample to be contained by the set of built subsamples. "
				+ "For more information, see:\n\n"
				+ getTechnicalInformation().toString();
	}

	/**
	 * Returns an instance of a TechnicalInformation object, containing 
	 * detailed information about the technical background of this class,
	 * e.g., paper reference or book this class is based on.
	 * 
	 * @return the technical information about this class
	 */
	public TechnicalInformation getTechnicalInformation() {
		TechnicalInformation 	result;

		result = new TechnicalInformation(Type.ARTICLE);
		result.setValue(Field.AUTHOR, "Jesús M. Pérez and Javier Muguerza and Olatz Arbelaitz and Ibai Gurrutxaga and José I. Martí­n");
		result.setValue(Field.YEAR, "2007");
		result.setValue(Field.TITLE, "Combining multiple class distribution modified subsamples in a single tree");
		result.setValue(Field.JOURNAL, "Pattern Recognition Letters");
		result.setValue(Field.VOLUME, "28");
		result.setValue(Field.NUMBER, "4");
		result.setValue(Field.PAGES, "414-422");
		result.setValue(Field.URL, "http://dx.doi.org/10.1016/j.patrec.2006.08.013");

		TechnicalInformation additional = new TechnicalInformation(Type.ARTICLE);
		additional.setValue(Field.AUTHOR, "Igor Ibarguren and Jesús M. Pérez and Javier Muguerza and Ibai Gurrutxaga and Olatz Arbelaitz");
		additional.setValue(Field.YEAR, "2015");
		additional.setValue(Field.TITLE, "Coverage-based resampling: Building robust consolidated decision trees");
		additional.setValue(Field.JOURNAL, "Knowledge Based Systems");
		additional.setValue(Field.VOLUME, "79");
		additional.setValue(Field.PAGES, "51-67");
		additional.setValue(Field.URL, "http://dx.doi.org/10.1016/j.knosys.2014.12.023");
		result.add(additional);

		return result;
	}

	/**
	 * Generates the classifier.
	 * (Implements the original CTC algorithm, so it
	 *  does not implement the options binarySplits and reducedErrorPruning of J48,
	 *  only what is based on C4.5 algorithm)
	 *
	 * @param instances the data to train the classifier with
	 * @throws Exception if classifier can't be built successfully
	 */
	public void buildClassifier(Instances instances) 
			throws Exception {

		// Some checks based on weka.classifiers.trees.J48.buildClassifier(Instances)
		if ((m_unpruned) && (!m_subtreeRaising)) {
			throw new Exception("Subtree raising does not need to be unset for unpruned trees!");
		}
		if ((m_unpruned) && (m_CF != 0.25f)) {
			throw new Exception("It does not make sense to change the confidence for an unpruned tree!");
		}
		if ((m_CF <= 0) || (m_CF >= 1)) {
			throw new Exception("Confidence has to be greater than zero and smaller than one!");
		}
		// can classifier tree handle the data?
		getCapabilities().testWithFail(instances);

		ModelSelection modSelection;
		// TODO Implement the option binarySplits of J48
		modSelection = new C45ConsolidatedModelSelection(m_minNumObj, instances, m_useMDLcorrection, m_doNotMakeSplitPointActualValue);
		// TODO Implement the option reducedErrorPruning of J48
		m_root = new C45ConsolidatedPruneableClassifierTree(modSelection, !m_unpruned,
				m_CF, m_subtreeRaising, !m_noCleanup, m_collapseTree);

		// remove instances with missing class before generating samples
		instances = new Instances(instances);
		instances.deleteWithMissingClass();

		//Generate as many samples as the number of samples with the given instances
		Instances[] samplesVector = generateSamples(instances);   
		//if (m_Debug)
		//	printSamplesVector(samplesVector);

		((C45ConsolidatedPruneableClassifierTree)m_root).buildClassifier(instances, samplesVector);

		((C45ModelSelection) modSelection).cleanup();
	}

	/**
	 * Generate as many samples as the number of samples based on Resampling Method parameters
	 * 
	 * @param instances the training data which will be used to generate the sample set
	 * @return Instances[] the vector of generated samples
	 * @throws Exception if something goes wrong
	 */
	protected Instances[] generateSamples(Instances instances) throws Exception {
		Instances[] samplesVector = null;
		// can classifier tree handle the data?
		getCapabilities().testWithFail(instances);

		// remove instances with missing class
		InstancesConsolidated instancesWMC = new InstancesConsolidated(instances);
		instancesWMC.deleteWithMissingClass();
		if (m_Debug) {
			System.out.println("=== Generation of the set of samples ===");
			System.out.println(toStringResamplingMethod());
		}
		/** Original sample size */
		int dataSize = instancesWMC.numInstances();
		if(dataSize==0)
			System.err.println("Original data size is 0! Handle zero training instances!");
		else
			if (m_Debug)
				System.out.println("Original data size: " + dataSize);
		/** Size of samples(bags) to be generated */
		int bagSize = 0;

		// Some checks done in set-methods
		//@ requires  0 <= m_RMnumberSamples 
		//@ requires -2 <= m_RMbagSizePercent && m_RMbagSizePercent <= 100 
		//@ requires -2 <= m_RMnewDistrMinClass && m_RMnewDistrMinClass < 100
		if(m_RMbagSizePercent >= 0 ){
			bagSize =  dataSize * m_RMbagSizePercent / 100;
			if(bagSize==0)
				System.err.println("Size of samples is 0 (" + m_RMbagSizePercent + "% of " + dataSize
						+ ")! Handle zero training instances!");
		} else if (m_RMnewDistrMinClass < 0) { // stratified OR free
			throw new Exception("Size of samples, m_RMbagSizePercent, (" + m_RMbagSizePercent + 
					") has to be between 0 and 100, when m_RMnewDistrMinClass < 0 (stratified or free)!!!");
		}

		Random random; 
		if (dataSize == 0) // To be OK when testing to Handle zero training instances!
			random = new Random(m_Seed);
		else
			random = instancesWMC.getRandomNumberGenerator(m_Seed);

		// Generate the vector of samples with the given parameters
		// TODO Set the different options to generate the samples like a filter and then use it here.  
		if(m_RMnewDistrMinClass == (float)-2)
			// stratified: Maintains the original class distribution in the new samples
			samplesVector = generateStratifiedSamples(instancesWMC, dataSize, bagSize, random);
		else if (m_RMnewDistrMinClass == (float)-1)
			// free: It doesn't take into account the original class distribution
			samplesVector = generateFreeDistrSamples(instancesWMC, dataSize, bagSize, random);
		else
			// RMnewDistrMinClass is between 0 and 100: Changes the class distribution to the indicated value
			samplesVector = generateSamplesChangingMinClassDistr(instancesWMC, dataSize, bagSize, random);
		if (m_Debug)
			System.out.println("=== End of Generation of the set of samples ===");
		return samplesVector;
	}
	
	

	/**
	 * Generate a set of stratified samples
	 * 
	 * @param instances the training data which will be used to generate the sample set
	 * @param dataSize Size of original sample (instances)
	 * @param bagSize Size of samples(bags) to be generated
	 * @param random a random number generator
	 * @return Instances[] the vector of generated samples
	 * @throws Exception if something goes wrong
	 */
	private Instances[] generateStratifiedSamples(
			InstancesConsolidated instances, int dataSize, int bagSize, Random random) throws Exception{
		int numClasses = instances.numClasses();
		// Get the classes
		InstancesConsolidated[] classesVector =  instances.getClasses();
		// What is the minority class?
		/** Vector containing the size of each class */
		int classSizeVector[] = instances.getClassesSize(classesVector);
		/** Index of the minority class in the original sample */
		int iMinClass = Utils.minIndex(classSizeVector);
		if (m_Debug)
			instances.printClassesInformation(dataSize , iMinClass, classSizeVector);

		// Determine the sizes of each class in the new samples
		/** Vector containing the size of each class in the new samples */
		int newClassSizeVector[] = new int [numClasses];
		// Check the bag size
		int bagSizePercent;
		if((dataSize == bagSize) && !m_RMreplacement){
			System.out.println("It doesn't make sense that the original sample's size and " +
					"the size of samples to be generated are the same without using replacement" +
					"because all the samples will be entirely equal!!!\n" +
					m_bagSizePercentToReduce + "% will be used as the bag size percentage!!!");
			bagSizePercent = m_bagSizePercentToReduce;
			bagSize =  dataSize * m_bagSizePercentToReduce / 100;
		}
		else
			bagSizePercent = m_RMbagSizePercent;
		/** Partial bag size */
		int localBagSize = 0;
		for(int iClass = 0; iClass < numClasses; iClass++)
			if(iClass != iMinClass){
				/** Value for the 'iClass'-th class size of the samples to be generated */
				int newClassSize = Utils.round(classSizeVector[iClass] * (double)bagSizePercent / 100);
				newClassSizeVector[iClass] = newClassSize;
				localBagSize += newClassSize;
			}
		/** Value for the minority class size of the samples to be generated */
		// (Done in this way to know the exact size of the minority class in the generated samples)
		newClassSizeVector[iMinClass] = bagSize - localBagSize;
		if (m_Debug) {
			System.out.println("New bag size: " + bagSize);
			System.out.println("Classes sizes of the new bag:");
			for (int iClass = 0; iClass < numClasses; iClass++){
				System.out.print(newClassSizeVector[iClass]);
				if(iClass < numClasses - 1)
					System.out.print(", ");
			}
			System.out.println("");
		}
		// Determine the size of samples' vector; the number of samples
		int numberSamples;
		/** Calculate the ratio of the sizes for each class between the sample and the subsample */
		double bagBySampleClassRatioVector[] = new double[numClasses];
		for(int iClass = 0; iClass < numClasses; iClass++)
			if (classSizeVector[iClass] > 0)
				bagBySampleClassRatioVector[iClass] = newClassSizeVector[iClass] / (double)classSizeVector[iClass];
			else // The size of the class is 0
				// This class won't be selected
				bagBySampleClassRatioVector[iClass] = Double.MAX_VALUE;
		if(m_RMnumberSamplesHowToSet == NumberSamples_BasedOnCoverage) {
			// The number of samples depends on the coverage to be guaranteed for the most disfavored class.
			double coverage = m_RMnumberSamples / (double)100;
			/** Calculate the most disfavored class in respect of coverage */
			int iMostDisfavorClass = Utils.minIndex(bagBySampleClassRatioVector);
			if (m_Debug) {
				System.out.println("Ratio bag:sample by each class:");
				System.out.println("(*) The most disfavored class based on coverage");
				for (int iClass = 0; iClass < numClasses; iClass++){
					System.out.print(Utils.doubleToString(bagBySampleClassRatioVector[iClass],2));
					if(iClass == iMostDisfavorClass)
						System.out.print("(*)");
					if(iClass < numClasses - 1)
						System.out.print(", ");
				}
				System.out.println("");
			}
			if(m_RMreplacement)
				numberSamples = (int) Math.ceil((-1) * Math.log(1 - coverage) / 
						bagBySampleClassRatioVector[iMostDisfavorClass]);
			else
				numberSamples = (int) Math.ceil(Math.log(1 - coverage) / 
						Math.log(1 - bagBySampleClassRatioVector[iMostDisfavorClass]));
			System.out.println("The number of samples to guarantee at least a coverage of " +
					Utils.doubleToString(100*coverage,0) + "% is " + numberSamples + ".");
			m_numberSamplesByCoverage = numberSamples;
			if (numberSamples < 3){
				numberSamples = 3;
				System.out.println("(*) Forced the number of samples to be 3!!!");
				m_stExceptionalSituationsMessage += " (*) Forced the number of samples to be 3!!!\n";
			}
		} else // m_RMnumberSamplesHowToSet == NumberSamples_FixedValue 
			// The number of samples has been set by parameter
			numberSamples = (int)m_RMnumberSamples;

		// Calculate the true coverage achieved
		m_trueCoverage = (double)0.0;
		for (int iClass = 0; iClass < numClasses; iClass++){
			double trueCoverageByClass;
			if(classSizeVector[iClass] > 0){
				if(m_RMreplacement)
					trueCoverageByClass = 1 - Math.pow(Math.E, (-1) * bagBySampleClassRatioVector[iClass] * numberSamples);
				else
					trueCoverageByClass = 1 - Math.pow((1 - bagBySampleClassRatioVector[iClass]), numberSamples);
			} else
				trueCoverageByClass = (double)0.0;
			double ratioClassDistr = classSizeVector[iClass] / (double)dataSize;
			m_trueCoverage += ratioClassDistr * trueCoverageByClass;
		}

		// Set the size of the samples' vector 
		Instances[] samplesVector = new Instances[numberSamples];

		// Generate the vector of samples 
		for(int iSample = 0; iSample < numberSamples; iSample++){
			InstancesConsolidated bagData = null;
			InstancesConsolidated bagClass = null;
			for(int iClass = 0; iClass < numClasses; iClass++){
				// Extract instances of the iClass-th class
				if(m_RMreplacement)
					bagClass = new InstancesConsolidated(classesVector[iClass].resampleWithWeights(random));
				else
					bagClass = new InstancesConsolidated(classesVector[iClass]);
				// Shuffle the instances
				bagClass.randomize(random);
				if (newClassSizeVector[iClass] < classSizeVector[iClass]) {
					InstancesConsolidated newBagData = new InstancesConsolidated(bagClass, 0, newClassSizeVector[iClass]);
					bagClass = newBagData;
					newBagData = null;
				}
				if(bagData == null)
					bagData = bagClass;
				else
					bagData.add(bagClass);
				bagClass = null;
			}
			// Shuffle the instances
			bagData.randomize(random);
			samplesVector[iSample] = (Instances)bagData;
			bagData = null;
		}
		classesVector = null;
		classSizeVector = null;
		newClassSizeVector = null;

		return samplesVector;
	}
	
	

	/**
	 * Generate a set of samples without taking the class distribution into account
	 * (like in the meta-classifier Bagging)
	 * 
	 * @param instances the training data which will be used to generate the sample set
	 * @param dataSize Size of original sample (instances)
	 * @param bagSize Size of samples(bags) to be generated
	 * @param random a random number generator
	 * @return Instances[] the vector of generated samples
	 * @throws Exception if something goes wrong
	 */
	private Instances[] generateFreeDistrSamples(
			InstancesConsolidated instances, int dataSize, int bagSize, Random random) throws Exception{
		// Check the bag size
		if((dataSize == bagSize) && !m_RMreplacement){
			System.out.println("It doesn't make sense that the original sample's size and " +
					"the size of samples to be generated are the same without using replacement" +
					"because all the samples will be entirely equal!!!\n" +
					m_bagSizePercentToReduce + "% will be used as the bag size percentage!!!");
			bagSize =  dataSize * m_bagSizePercentToReduce / 100;
		}
		if (m_Debug)
			System.out.println("New bag size: " + bagSize);
		// Determine the size of samples' vector; the number of samples
		int numberSamples;
		double bagBySampleRatio = bagSize / (double) dataSize;
		if(m_RMnumberSamplesHowToSet == NumberSamples_BasedOnCoverage) {
			// The number of samples depends on the coverage to be guaranteed for the most disfavored class.
			double coverage = m_RMnumberSamples / (double)100;
			if(m_RMreplacement)
				numberSamples = (int) Math.ceil((-1) * Math.log(1 - coverage) / 
						bagBySampleRatio);
			else
				numberSamples = (int) Math.ceil(Math.log(1 - coverage) / 
						Math.log(1 - bagBySampleRatio));
			System.out.println("The number of samples to guarantee at least a coverage of " +
					Utils.doubleToString(100*coverage,0) + "% is " + numberSamples + ".");
			m_numberSamplesByCoverage = numberSamples;
			if (numberSamples < 3){
				numberSamples = 3;
				System.out.println("(*) Forced the number of samples to be 3!!!");
				m_stExceptionalSituationsMessage += " (*) Forced the number of samples to be 3!!!\n";
			}
		} else // m_RMnumberSamplesHowToSet == NumberSamples_FixedValue
			// The number of samples has been set by parameter
			numberSamples = (int)m_RMnumberSamples;

		// Calculate the true coverage achieved
		if(m_RMreplacement)
			m_trueCoverage = 1 - Math.pow(Math.E, (-1) * bagBySampleRatio * numberSamples);
		else
			m_trueCoverage = 1 - Math.pow((1 - bagBySampleRatio), numberSamples);

		// Set the size of the samples' vector 
		Instances[] samplesVector = new Instances[numberSamples];

		// Generate the vector of samples 
		for(int iSample = 0; iSample < numberSamples; iSample++){
			Instances bagData = null;
			if(m_RMreplacement)
				bagData = new Instances(instances.resampleWithWeights(random));
			else
				bagData = new Instances(instances);
			// Shuffle the instances
			bagData.randomize(random);
			if (bagSize < dataSize) {
				Instances newBagData = new Instances(bagData, 0, bagSize);
				bagData = newBagData;
				newBagData = null;
			}
			samplesVector[iSample] = bagData;
			bagData = null;
		}
		return samplesVector;
	}
	
	
	

	/**
	 * Generate a set of samples changing the distribution of the minority class
	 * 
	 * @param instances the training data which will be used to generate the sample set
	 * @param dataSize Size of original sample (instances)
	 * @param bagSize Size of samples(bags) to be generated
	 * @param random a random number generator
	 * @return Instances[] the vector of generated samples
	 * @throws Exception if something goes wrong
	 */
	private Instances[] generateSamplesChangingMinClassDistr(
			InstancesConsolidated instances, int dataSize, int bagSize, Random random) throws Exception{
		int numClasses = instances.numClasses();
		// Some checks
		if((numClasses > 2) && (m_RMnewDistrMinClass != (float)50.0))
			throw new Exception("In the case of multi-class datasets, the only posibility to change the distribution of classes is to balance them!!!\n" +
					"Use the special value '50.0' in <distribution minority class> for this purpose!!!");
		// TODO Generalize the process to multi-class datasets to set different new values of distribution for each classs.
		// Some checks done in set-methods
		//@ requires m_RMreplacement = false 
		// TODO Accept replacement

		// Get the classes
		InstancesConsolidated[] classesVector =  instances.getClasses();

		// What is the minority class?
		/** Vector containing the size of each class */
		int classSizeVector[] = instances.getClassesSize(classesVector);
		/** Index of the minority class in the original sample */
		int iMinClass, i_iMinClass;
		/** Prevent the minority class from being empty (we hope there is one non-empty!) */
		int iClassSizeOrdVector[] = Utils.sort(classSizeVector);
		for(i_iMinClass = 0; ((i_iMinClass < numClasses) && (classSizeVector[iClassSizeOrdVector[i_iMinClass]] == 0)); i_iMinClass++);
		if(i_iMinClass < numClasses)
			iMinClass = iClassSizeOrdVector[i_iMinClass];
		else // To be OK when testing to Handle zero training instances!
			iMinClass = 0;

		/** Index of the majority class in the original sample */
		int iMajClass = Utils.maxIndex(classSizeVector);
		/** Determines whether the original sample is balanced or not */
		boolean isBalanced = false;
		if (iMinClass == iMajClass){
			isBalanced = true;
			// If the sample is balanced, it is determined, by convention, that the majority class is the last one
			iMajClass = numClasses-1;
		}
		if (m_Debug)
			instances.printClassesInformation(dataSize , iMinClass, classSizeVector);

		/** Distribution of the minority class in the original sample */
		float distrMinClass;
		if (dataSize == 0)
			distrMinClass = (float)0;
		else
			distrMinClass = (float)100 * classSizeVector[iMinClass] / dataSize;

		/** Guarantee the minimum number of examples in each class based on m_minExamplesPerClassPercent */
		int minExamplesPerClass = (int) Math.ceil(dataSize * m_minExamplesPerClassPercent / (double)100.0) ;
		/** Guarantee to be at least m_minNumObj */
		if (minExamplesPerClass < m_minNumObj)
			minExamplesPerClass = m_minNumObj;
		if (m_Debug)
			System.out.println("Minimum number of examples to be guaranteed in each class: " + minExamplesPerClass);
		for(int iClass = 0; iClass < numClasses; iClass++){
			if((classSizeVector[iClass] < minExamplesPerClass) && // if number of examples is smaller than the minimum
					(classSizeVector[iClass] > 0)){					// but, at least, it has to exist any example.
				// Oversample the class ramdonly
				System.out.println("The " + iClass + "-th class has too few examples (" + classSizeVector[iClass]+ ")!!!\n" +
						"It will be oversampled ranmdoly up to " + minExamplesPerClass + "!!!");
				m_stExceptionalSituationsMessage += " (*) Forced the " + iClass + "-th class to be oversampled!!!\n";
				// based on the code of the function 'resample(Random)' of the class 'Instances'
				InstancesConsolidated bagClass = classesVector[iClass];
				while (bagClass.numInstances() < minExamplesPerClass) {
					bagClass.add(classesVector[iClass].instance(random.nextInt(classSizeVector[iClass])));
				}
				// Update the vectors with classes' information and the new data size
				dataSize = dataSize - classSizeVector[iClass] + minExamplesPerClass; 
				classesVector[iClass] = bagClass;
				classSizeVector[iClass] = minExamplesPerClass;
			}
		}

		/** Maximum values for classes' size on the samples to be generated taking RMnewDistrMinClass into account
		 *   and without using replacement */
		int maxClassSizeVector[] = new int[numClasses];
		if (numClasses == 2){
			// the dataset is two-class
			if(m_RMnewDistrMinClass > distrMinClass){
				// Maintains the whole minority class
				maxClassSizeVector[iMinClass] = classSizeVector[iMinClass];
				maxClassSizeVector[iMajClass] = Utils.round(classSizeVector[iMinClass] * (100 - m_RMnewDistrMinClass) / m_RMnewDistrMinClass);
			} else {
				// Maintains the whole majority class
				maxClassSizeVector[iMajClass] = classSizeVector[iMajClass];
				maxClassSizeVector[iMinClass] = Utils.round(classSizeVector[iMajClass] * m_RMnewDistrMinClass / (100 - m_RMnewDistrMinClass));
			}
		} else {
			// the dataset is multi-class
			/** The only accepted option is to change the class distribution is to balance the samples */
			for(int iClass = 0; iClass < numClasses; iClass++)
				maxClassSizeVector[iClass] = classSizeVector[iMinClass];
		}

		// Determine the sizes of each class in the new samples
		/** Vector containing the size of each class in the new samples */
		int newClassSizeVector[] = new int[numClasses];
		/** Determines whether the size of samples to be generated will be forced to be reduced in exceptional situations */
		boolean forceToReduceSamplesSize = false;
		if(m_RMbagSizePercent == -2){
			// maxSize : Generate the biggest samples according to the indicated distribution (RMnewDistrMinClass),
			//  that is, maintaining the whole minority (majority) class
			if (numClasses == 2){
				// the dataset is two-class
				if(Utils.eq(m_RMnewDistrMinClass, distrMinClass)){
					System.out.println("It doesn't make sense that the original distribution and " +
							"the distribution to be changed (RMnewDistrMinClass) are the same and " +
							"the size of samples to be generated is maximum (RMbagSizePercent=-2) " +
							"(without using replacement) " +
							"because all the samples will be entirely equal!!!\n" +
							m_bagSizePercentToReduce + "% will be used as the bag size percentage!!!");
					forceToReduceSamplesSize = true;
				}
			} else
				// the dataset is multi-class
				if (isBalanced){
					System.out.println("In the case of multi-class datasets, if the original sample is balanced, " +
							"it doesn't make sense that " +
							"the size of samples to be generated is maximum (RMbagSizePercent=-2) " +
							"(without using replacement) " +
							"because all the samples will be entirely equal!!!\n" +
							m_bagSizePercentToReduce + "% will be used as the bag size percentage!!!");
					forceToReduceSamplesSize = true;
				}
			if(!forceToReduceSamplesSize){
				bagSize = 0;
				for(int iClass = 0; iClass < numClasses; iClass++)
					if (classSizeVector[iClass] == 0)
						newClassSizeVector[iClass] = 0;
					else {
						newClassSizeVector[iClass] = maxClassSizeVector[iClass];
						bagSize += maxClassSizeVector[iClass];
					}
			}
		} else {
			if (m_RMbagSizePercent == -1)
				// sizeOfMinClass: the generated samples will have the same size that the minority class
				bagSize = classSizeVector[iMinClass];
			else
				// m_RMbagSizePercent is between 0 and 100. bagSize is already set.
				if(dataSize == bagSize){
					System.out.println("It doesn't make sense that the original sample's size and " +
							"the size of samples to be generated are the same" +
							"(without using replacement) " +
							"because all the samples will be entirely equal!!!\n" +
							m_bagSizePercentToReduce + "% will be used as the bag size percentage!!!");
					forceToReduceSamplesSize = true;
				}
		}
		if((m_RMbagSizePercent != -2) || forceToReduceSamplesSize)
		{
			if (numClasses == 2){
				// the dataset is two-class
				if(forceToReduceSamplesSize){
					bagSize =  dataSize * m_bagSizePercentToReduce / 100;
					m_stExceptionalSituationsMessage += " (*) Forced to reduce the size of the generated samples!!!\n";
				}
				newClassSizeVector[iMinClass] = Utils.round(m_RMnewDistrMinClass * bagSize / 100);
				newClassSizeVector[iMajClass] = bagSize - newClassSizeVector[iMinClass];
			} else {
				// the dataset is multi-class
				/** The only accepted option is to change the class distribution is to balance the samples */
				/** All the classes will have the same size, classSize, based on bagSizePercent applied 
				 *  on minority class, that is, the generated samples will be bigger than expected, 
				 *  neither sizeofMinClass nor original sample's size by bagSizePercent. Otherwise,
				 *  the classes of the samples would be too unpopulated */
				int bagSizePercent;
				if (m_RMbagSizePercent == -1)
					/** sizeOfMinClass (-1) is a special case, where the bagSizePercent to be applied will be
					 *  the half of the minority class, the same that it would be achieved if the dataset was two-class */
					bagSizePercent = 50;
				else
					if (forceToReduceSamplesSize)
						bagSizePercent = m_bagSizePercentToReduce;
					else
						bagSizePercent = m_RMbagSizePercent;
				int classSize = (int)(bagSizePercent * classSizeVector[iMinClass] / (float)100);
				bagSize = 0;
				for(int iClass = 0; iClass < numClasses; iClass++)
					if (classSizeVector[iClass] == 0)
						newClassSizeVector[iClass] = 0;
					else {
						newClassSizeVector[iClass] = classSize;
						bagSize += classSize;
					}
			}
		}

		if (m_Debug) {
			System.out.println("New bag size: " + bagSize);
			System.out.println("New minority class size: " + newClassSizeVector[iMinClass] + " (" + (int)(newClassSizeVector[iMinClass] / (double)bagSize * 100) + "%)");
			System.out.print("New majority class size: " + newClassSizeVector[iMajClass]);
			if (numClasses > 2)
				System.out.print(" (" + (int)(newClassSizeVector[iMajClass] / (double)bagSize * 100) + "%)");
			System.out.println();
		}
		// Some checks
		for(int iClass = 0; iClass < numClasses; iClass++)
			if(newClassSizeVector[iClass] > classSizeVector[iClass])
				throw new Exception("There aren't enough instances of the " + iClass + 
						"-th class (" +	classSizeVector[iClass] +
						") to extract " + newClassSizeVector[iClass] +
						" for the new samples whithout replacement!!!");

		// Determine the size of samples' vector; the number of samples
		int numberSamples;
		/** Calculate the ratio of the sizes for each class between the sample and the subsample */
		double bagBySampleClassRatioVector[] = new double[numClasses];
		for(int iClass = 0; iClass < numClasses; iClass++)
			if (classSizeVector[iClass] > 0)
				bagBySampleClassRatioVector[iClass] = newClassSizeVector[iClass] / (double)classSizeVector[iClass];
			else // The size of the class is 0
				// This class won't be selected
				bagBySampleClassRatioVector[iClass] = Double.MAX_VALUE;
		if(m_RMnumberSamplesHowToSet == NumberSamples_BasedOnCoverage) {
			// The number of samples depends on the coverage to be guaranteed for the most disfavored class.
			double coverage = m_RMnumberSamples / (double)100;
			/** Calculate the most disfavored class in respect of coverage */
			int iMostDisfavorClass = Utils.minIndex(bagBySampleClassRatioVector);
			if (m_Debug) {
				System.out.println("Ratio bag:sample by each class:");
				System.out.println("(*) The most disfavored class based on coverage");
				for (int iClass = 0; iClass < numClasses; iClass++){
					System.out.print(Utils.doubleToString(bagBySampleClassRatioVector[iClass],2));
					if(iClass == iMostDisfavorClass)
						System.out.print("(*)");
					if(iClass < numClasses - 1)
						System.out.print(", ");
				}
				System.out.println("");
			}
			numberSamples = (int) Math.ceil(Math.log(1 - coverage) / 
					Math.log(1 - bagBySampleClassRatioVector[iMostDisfavorClass]));
			System.out.println("The number of samples to guarantee at least a coverage of " +
					Utils.doubleToString(100*coverage,0) + "% is " + numberSamples + ".");
			m_numberSamplesByCoverage = numberSamples;
			if (numberSamples < 3){
				numberSamples = 3;
				System.out.println("(*) Forced the number of samples to be 3!!!");
				m_stExceptionalSituationsMessage += " (*) Forced the number of samples to be 3!!!\n";
			}
		} else // m_RMnumberSamplesHowToSet == NumberSamples_FixedValue
			// The number of samples has been set by parameter
			numberSamples = (int)m_RMnumberSamples;

		// Calculate the true coverage achieved
		m_trueCoverage = (double)0.0;
		for (int iClass = 0; iClass < numClasses; iClass++){
			double trueCoverageByClass;
			if(classSizeVector[iClass] > 0){
				if(m_RMreplacement)
					trueCoverageByClass = 1 - Math.pow(Math.E, (-1) * bagBySampleClassRatioVector[iClass] * numberSamples);
				else
					trueCoverageByClass = 1 - Math.pow((1 - bagBySampleClassRatioVector[iClass]), numberSamples);
			} else
				trueCoverageByClass = (double)0.0;
			double ratioClassDistr = classSizeVector[iClass] / (double)dataSize;
			m_trueCoverage += ratioClassDistr * trueCoverageByClass;
		}

		// Set the size of the samples' vector 
		Instances[] samplesVector = new Instances[numberSamples];

		// Generate the vector of samples 
		for(int iSample = 0; iSample < numberSamples; iSample++){
			InstancesConsolidated bagData = null;
			InstancesConsolidated bagClass = null;

			for (int iClass = 0; iClass < numClasses; iClass++)
				if (classSizeVector[iClass] > 0){
					// Extract instances of the i-th class
					bagClass = new InstancesConsolidated(classesVector[iClass]);
					// Shuffle the instances
					bagClass.randomize(random);
					if (newClassSizeVector[iClass] < classSizeVector[iClass]) {
						InstancesConsolidated newBagData = new InstancesConsolidated(bagClass, 0, newClassSizeVector[iClass]);
						bagClass = newBagData;
						newBagData = null;
					}
					// Add the bagClass (i-th class) to bagData
					if (bagData == null)
						bagData = bagClass;
					else
						bagData.add(bagClass);
					bagClass = null;
				}
			// Shuffle the instances
			if (bagData == null) // To be OK when testing to Handle zero training instances!
				bagData = instances;
			else
				bagData.randomize(random);
			samplesVector[iSample] = (Instances)bagData;
			bagData = null;
		}
		classesVector = null;
		classSizeVector = null;
		maxClassSizeVector = null;
		newClassSizeVector = null;

		return samplesVector;
	}

	
	
	/**
	 * Print the generated samples. Only for testing purposes. 
	 *
	 * @param samplesVector the vector of samples
	 */
	protected void printSamplesVector(Instances[] samplesVector){

		for(int iSample=0; iSample<samplesVector.length; iSample++){
			System.out.println("==== SAMPLE " + iSample + " ====");
			System.out.println(samplesVector[iSample]);
			System.out.println(" ");
		}
	}

	/**
	 * Returns an enumeration describing the available options.
	 *
	 * Valid options are:<br/>
	 * 
	 * J48 options<br/>
	 * =============<br/>
	 *
	 * Options to set the Resampling Method (RM) for the generation of samples
	 *  to use in the consolidation process
	 * ============================================================================ 
	 * <pre>-RM-C
	 * Determines the way to set the number of samples to be generated will be based on
	 * a coverage value as a percentage. In the case this option is not set, the number of samples
	 * will be determined using a fixed value.
	 * (set by default)</pre>
	 * 
	 * <pre>-RM-N &lt;number of samples&gt;
	 * Number of samples to be generated for the use in the construction of the consolidated tree.
	 * It can be set as a fixed value or based on a coverage value as a percentage, when -RM-C option
	 * is used, which guarantees the number of samples necessary to adequately cover the examples 
	 * of the original sample
	 * (Default 5 for a fixed value or 99% for the case based on a coverage value)</pre>
	 * 
	 * <pre>-RM-R
	 * Determines whether or not replacement is used when generating the samples.
	 * (Default: false)</pre>
	 * 
	 * <pre>-RM-B percentage
	 * Size of each sample(bag), as a percentage of the training set size.
	 * Combined with the option &lt;distribution minority class&gt; accepts:
	 *  * -1 (sizeOfMinClass): The size of the minority class
	 *  * -2 (maxSize): Maximum size taking &lt;distribution minority class&gt; into account
	 *              and using no replacement
	 * (Default: -2(maxSize))</pre>
	 * 
	 * <pre>-RM-D distribution minority class
	 * Determines the new value of the distribution of the minority class, if we want to change it.
	 * It can be one of the following values:
	 *  * A value between 0 and 100 to change the portion of minority class instances in the new samples
	 *    (If the dataset is multi-class, only the special value 50.0 will be accepted to balance the classes)
	 *  * -1 (free): Works with the instances without taking their class into account
	 *  * -2 (stratified): Maintains the original class distribution in the new samples
	 * (Default: -1(free))</pre>
	 * 
	 * @return an enumeration of all the available options.
	 */
	public Enumeration<Option> listOptions() {

		Vector<Option> newVector = new Vector<Option>();

		// J48 options
		// ===========
		Enumeration<Option> en;
		en = super.listOptions();
		while (en.hasMoreElements())
			newVector.addElement((Option) en.nextElement());

		// Options to set the Resampling Method (RM) for the generation of samples
		//  to use in the consolidation process
		// =========================================================================
		newVector.
		addElement(new Option("\tSet the number of samples to be generated based on a coverage value\n" +
				"\tas a percentage (by default)",
				"RM-C", 0, "-RM-C"));
		newVector.
		addElement(new Option("\tNumber of samples to be generated for the use in the construction of the\n" +
				"\tconsolidated tree.\n" +
				"\tIt can be set as a fixed value or based on a coverage value as a percentage, \n" +
				"\twhen -RM-C option is used, which guarantees the number of samples necessary \n" +
				"\tto adequately cover the examples of the original sample.\n" +
				"\t(default: 5 for a fixed value and \n" +
				"\t " + Utils.doubleToString(m_coveragePercent,0) + "% for the case based on a coverage value)",
				"RM-N", 1, "-RM-N <Number of samples>"));
		newVector.
		addElement(new Option("\tUse replacement to generate the set of samples\n" +
				"\t(default false)",
				"RM-R", 0, "-RM-R"));
		newVector.
		addElement(new Option("\tSize of each sample(bag), as a percentage of the training set size.\n" +
				"\tCombined with the option <distribution minority class> accepts:\n" +
				"\t * -1 (sizeOfMinClass): The size of the minority class\n" +
				"\t * -2 (maxSize): Maximum size taking <distribution minority class>\n" +
				"\t             into account and using no replacement\n" +
				"\t(default -2(maxSize))",
				"RM-B", 1, "-RM-B <Size of each sample(%)>"));
		newVector.
		addElement(new Option(
				"\tDetermines the new value of the distribution of the minority class.\n" +
						"\tIt can be one of the following values:\n" +
						"\t * A value between 0 and 100 to change the portion of minority class\n" +
						"\t              instances in the new samples\n" +
						"\t   (If the dataset is multi-class, only the special value 50.0 will\n" +
						"\t              be accepted to balance the classes)\n" +
						"\t * -1 (free): Works with the instances without taking their class\n" +
						"\t              into account\n" +
						"\t * -2 (stratified): Maintains the original class distribution in the\n" +
						"\t              new samples\n" +
						"\t(default 50.0)",
						"RM-D", 1, "-RM-D <distribution minority class>"));

		return newVector.elements();
	}

	/**
	 * Parses a given list of options.
	 * 
   <!-- options-start -->
	 * Valid options are: <p/>
	 * 
	 * J48 options<br/>
	 * =============<br/>
	 *
	 * Options to set the Resampling Method (RM) for the generation of samples
	 *  to use in the consolidation process
	 * ============================================================================ 
	 * <pre> -RM-C
	 *  Determines the way to set the number of samples to be generated will be based on
	 *  a coverage value as a percentage. In the case this option is not set, the number of samples
	 *  will be determined using a fixed value. 
	 *  (set by default)</pre>
	 * 
	 * <pre> -RM-N &lt;number of samples&gt;
	 *  Number of samples to be generated for the use in the construction of the consolidated tree.
	 *  It can be set as a fixed value or based on a coverage value as a percentage, when -RM-C option
	 *  is used, which guarantees the number of samples necessary to adequately cover the examples 
	 *  of the original sample
	 *  (default 5 for a fixed value or 99% for the case based on a coverage value)</pre>
	 * 
	 * <pre> -RM-R
	 *  Determines whether or not replacement is used when generating the samples.
	 *  (default true)</pre>
	 * 
	 * <pre> -RM-B &lt;Size of each sample(&#37;)&gt;
	 *  Size of each sample(bag), as a percentage of the training set size.
	 *  Combined with the option &lt;distribution minority class&gt; accepts:
	 *  * -1 (sizeOfMinClass): The size of the minority class  
	 *  * -2 (maxSize): Maximum size taking &lt;distribution minority class&gt; into account
	 *  *           and using no replacement
	 *  (default -2(maxSize))</pre>
	 * 
	 * <pre> -RM-D &lt;distribution minority class&gt;
	 *  Determines the new value of the distribution of the minority class, if we want to change it.
	 *  It can be one of the following values:
	 *  * A value between 0 and 100 to change the portion of minority class instances in the new samples
	 *    (If the dataset is multi-class, only the special value 50.0 will be accepted to balance the classes)
	 *  * -1 (free): Works with the instances without taking their class into account
	 *  * -2 (stratified): Maintains the original class distribution in the new samples
	 *  (default 50.0)</pre>
	 * 
   <!-- options-end -->
	 *
	 * @param options the list of options as an array of strings
	 * @throws Exception if an option is not supported
	 */
	public void setOptions(String[] options) throws Exception {

		// Options to set the Resampling Method (RM) for the generation of samples
		//  to use in the consolidation process
		// =========================================================================
		if (Utils.getFlag("RM-C", options))
			setRMnumberSamplesHowToSet(new SelectedTag(NumberSamples_BasedOnCoverage, TAGS_WAYS_TO_SET_NUMBER_SAMPLES));
		else
			setRMnumberSamplesHowToSet(new SelectedTag(NumberSamples_FixedValue, TAGS_WAYS_TO_SET_NUMBER_SAMPLES));
		String RMnumberSamplesString = Utils.getOption("RM-N", options);
		if (RMnumberSamplesString.length() != 0) {
			setRMnumberSamples(Float.parseFloat(RMnumberSamplesString));
		}
		else {
			if (m_RMnumberSamplesHowToSet == NumberSamples_BasedOnCoverage)
				setRMnumberSamples((float)m_coveragePercent); // default: f(99% of coverage)
			else
				setRMnumberSamples((float)5.0);  // default: 5 samples
		}
		String RMbagSizePercentString = Utils.getOption("RM-B", options);
		if (RMbagSizePercentString.length() != 0)
			setRMbagSizePercent(Integer.parseInt(RMbagSizePercentString), false);
		else
			setRMbagSizePercent(-2, false); // default: maxSize
		String RMnewDistrMinClassString = Utils.getOption("RM-D", options);
		if (RMnewDistrMinClassString.length() != 0)
			setRMnewDistrMinClass(Float.parseFloat(RMnewDistrMinClassString), false);
		else
			setRMnewDistrMinClass((float)50.0, false);
		// Only checking the combinations of the three options RMreplacement, RMbagSizePercent and
		//  RMnewDistrMinClass when they all are set.
		setRMreplacement(Utils.getFlag("RM-R", options), true);
		// J48 options
		// ===========
		super.setOptions(options);
	}

	/**
	 * Gets the current settings of the Classifier.
	 *
	 * @return an array of strings suitable for passing to setOptions
	 */
	public String [] getOptions() {

		Vector<String> result = new Vector<String>();

		// J48 options
		// ===========
		String[] options = super.getOptions();
		for (int i = 0; i < options.length; i++)
			result.add(options[i]);
		// In J48 m_Seed is added only if m_reducedErrorPruning is true,
		// but in J48Consolidated it is necessary to the generation of the samples 
		result.add("-Q");
		result.add("" + m_Seed);

		// Options to set the Resampling Method (RM) for the generation of samples
		//  to use in the consolidation process
		// =========================================================================
		if (m_RMnumberSamplesHowToSet == NumberSamples_BasedOnCoverage)
			result.add("-RM-C");
		result.add("-RM-N");
		result.add(""+ m_RMnumberSamples);
		if (m_RMreplacement)
			result.add("-RM-R");
		result.add("-RM-B");
		result.add("" + m_RMbagSizePercent);
		result.add("-RM-D");
		result.add("" + m_RMnewDistrMinClass);

		return (String[]) result.toArray(new String[result.size()]);	  
	}

	/**
	 * Returns a description of the classifier.
	 * 
	 * @return a description of the classifier
	 */
	public String toString() {

		if (m_root == null)
			return "No classifier built";
		return "J48Consolidated " + (m_unpruned? "unpruned " : "") + "tree\n" + 
			toStringResamplingMethod() + m_root.toString();
	}

	/**
	 * Returns a description of the Resampling Method used in the consolidation process.
	 * 
	 * @return a description of the used Resampling Method (RM)
	 */
	public String toStringResamplingMethod() {
		String st;
		st = "[RM] N_S=";
		if (m_RMnumberSamplesHowToSet == NumberSamples_BasedOnCoverage){
			st += "f(" + Utils.doubleToString(m_RMnumberSamples,2) + "% of coverage)";
			if (m_numberSamplesByCoverage != 0)
				st += "=" + m_numberSamplesByCoverage;
		}
		else // m_RMnumberSamplesHowToSet == NumberSamples_FixedValue
			st += (int)m_RMnumberSamples;
		if (m_RMnewDistrMinClass == -2)
			st += " stratified";
		else if (m_RMnewDistrMinClass == -1)
			st += " free distribution";
		else {
			st += " %Min=";
			if (Utils.eq(m_RMnewDistrMinClass, (float)50))
				st += "balanced";
			else
				st += m_RMnewDistrMinClass;
		}
		st += " Size=";
		if (m_RMbagSizePercent == -2)
			st += "maxSize";
		else if (m_RMbagSizePercent == -1)
			st += "sizeOfMinClass";
		else
			st += m_RMbagSizePercent + "%";
		if (m_RMreplacement)
			st += " (with replacement)";
		else
			st += " (without replacement)";
		st += "\n";
		st += m_stExceptionalSituationsMessage;
		// Add the true coverage achieved
		st += "True coverage achieved: " + m_trueCoverage + "\n";
		// Add a separator
		char[] ch_line = new char[st.length()];
		for (int i = 0; i < ch_line.length; i++)
			ch_line[i] = '-';
		String line = String.valueOf(ch_line);
		line += "\n";
		st += line;
		return st;
	}

	/**
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String RMnumberSamplesHowToSetTipText() {
		return "Way to set the number of samples to be generated:\n" +
				" * using a fixed value which directly indicates the number of samples to be generated\n" +
				" * based on a coverage value as a percentage (by default)\n";
	}

	/**
	 * Get the value of RMnumberSamplesHowToSet.
	 *
	 * @return Value of RMnumberSamplesHowToSet.
	 */
	public SelectedTag getRMnumberSamplesHowToSet() {
		return new SelectedTag(m_RMnumberSamplesHowToSet,
				TAGS_WAYS_TO_SET_NUMBER_SAMPLES);
	}

	/**
	 * Set the value of RMnumberSamplesHowToSet. Values other than
	 * NumberSamples_FixedValue, or NumberSamples_BasedOnCoverage will be ignored.
	 *
	 * @param newWayToSetNumberSamples the way to set the number of samples to use
	 * @throws Exception if an option is not supported
	 */
	public void setRMnumberSamplesHowToSet(SelectedTag newWayToSetNumberSamples) throws Exception {
		if (newWayToSetNumberSamples.getTags() == TAGS_WAYS_TO_SET_NUMBER_SAMPLES) 
		{
			int newEvWay = newWayToSetNumberSamples.getSelectedTag().getID();

			if (newEvWay == NumberSamples_FixedValue || newEvWay == NumberSamples_BasedOnCoverage)
				m_RMnumberSamplesHowToSet = newEvWay;
			else 
				throw new IllegalArgumentException("Wrong selection type, value should be: "
						+ "between 1 and 2");
		}
	}

	/**
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String RMnumberSamplesTipText() {
		return "Number of samples to be generated for the use in the consolidation process (fixed value) or based on a coverage value as a %.\n" +
				" * if RMnumberSamplesHowToSet == " + (new SelectedTag(NumberSamples_FixedValue,TAGS_WAYS_TO_SET_NUMBER_SAMPLES)).getSelectedTag().getReadable() + "\n" +
				"    A positive value which directly indicates the number of samples to be generated\n" + 
				" * if RMnumberSamplesHowToSet == " + (new SelectedTag(NumberSamples_BasedOnCoverage,TAGS_WAYS_TO_SET_NUMBER_SAMPLES)).getSelectedTag().getReadable() + "\n" +
				"    A positive value as a percentage, the coverage value, which guarantees the number of samples necessary\n" +
				"    to adequately cover the examples of the original sample\n" +
				" (default: 5 for a fixed value or " + Utils.doubleToString(m_coveragePercent,0) + "% for the case based on a coverage value)";
	}

	/**
	 * Get the value of RMnumberSamples.
	 *
	 * @return Value of RMnumberSamples.
	 */
	public float getRMnumberSamples() {

		return m_RMnumberSamples;
	}

	/**
	 * Set the value of RMnumberSamples.
	 *
	 * @param v  Value to assign to RMnumberSamples.
	 * @throws Exception if an option is not supported
	 */
	public void setRMnumberSamples(float v) throws Exception {
		if (m_RMnumberSamplesHowToSet == NumberSamples_FixedValue) {
			if (v < 0)
				throw new Exception("Number of samples has to be greater than zero!");
			if (v == (float)0)
				System.err.println("Number of samples is 0. It doesn't make sense to build a consolidated tree without set of samples. Handle zero training instances!");
			if ((v == (float)1) || (v == (float)2))
				System.out.println("It doesn't make sense to build a consolidated tree with 1 or 2 samples, but it's possible!");
		} else { // m_RMnumberSamplesHowToSet == NumberSamples_BasedOnCoverage
			if (v < -1)
				throw new Exception("Coverage value has to be greater than zero!");
			if (v == (float)0)
				System.err.println("Coverage value is 0. It doesn't make sense to build a consolidated tree without set of samples. Handle zero training instances!");
		}
		m_RMnumberSamples = v;
	}

	/**
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String RMreplacementTipText() {
		return "Whether replacement is performed to generate the set of samples.";
	}

	/**
	 * Get the value of RMreplacement
	 *
	 * @return Value of RMreplacement
	 */
	public boolean getRMreplacement() {

		return m_RMreplacement;
	}

	/**
	 * Set the value of RMreplacement.
	 * Checks the combinations of the options RMreplacement, RMbagSizePercent and RMnewDistrMinClass
	 *  
	 * @param v  Value to assign to RMreplacement.
	 * @throws Exception if an option is not supported
	 */
	public void setRMreplacement(boolean v) throws Exception {

		setRMreplacement(v, true);
	}

	/**
	 * Set the value of RMreplacement, but, optionally,
	 * checks the combinations of the options RMreplacement, RMbagSizePercent and RMnewDistrMinClass.
	 * This makes possible only checking in the last call of the method setOptions().
	 *  
	 * @param v  Value to assign to RMreplacement.
	 * @param checkComb true to check some combinations of options
	 * @throws Exception if an option is not supported
	 */
	public void setRMreplacement(boolean v, boolean checkComb) throws Exception {

		if(checkComb)
			checkBagSizePercentAndReplacementAndNewDistrMinClassOptions(v, m_RMbagSizePercent, m_RMnewDistrMinClass);
		m_RMreplacement = v;
	}

	/**
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String RMbagSizePercentTipText() {
		return "Size of each sample(bag), as a percentage of the training set size/-1=sizeOfMinClass/-2=maxSize.\n" +
				"Combined with the option <distribution minority class>, RMnewDistrMinClass, accepts:\n" +
				" * -1 (sizeOfMinClass): The size of the minority class\n" +
				" * -2 (maxSize): Maximum size taking <distribution minority class> into account\n" +
				"             and using no replacement." +
				" (default: -2 (maxSize))";
	}

	/**
	 * Get the value of RMbagSizePercent.
	 *
	 * @return Value of RMbagSizePercent.
	 */
	public int getRMbagSizePercent() {

		return m_RMbagSizePercent;
	}

	/**
	 * Set the value of RMbagSizePercent.
	 * Checks the combinations of the options RMreplacement, RMbagSizePercent and RMnewDistrMinClass
	 *
	 * @param v  Value to assign to RMbagSizePercent.
	 * @throws Exception if an option is not supported
	 */
	public void setRMbagSizePercent(int v) throws Exception {

		setRMbagSizePercent(v, true);
	}

	/**
	 * Set the value of RMbagSizePercent, but, optionally,
	 * checks the combinations of the options RMreplacement, RMbagSizePercent and RMnewDistrMinClass.
	 * This makes possible only checking in the last call of the method setOptions().
	 *  
	 * @param v  Value to assign to RMbagSizePercent.
	 * @param checkComb true to check some combinations of options
	 * @throws Exception if an option is not supported
	 */
	public void setRMbagSizePercent(int v, boolean checkComb) throws Exception {

		if ((v < -2) || (v > 100))
			throw new Exception("Size of sample (%) has to be greater than zero and smaller " +
					"than or equal to 100 " +
					"(or combining with the option <distribution minority class> -1 for 'sizeOfMinClass' " +
					"or -2 for 'maxSize')!");
		else if (v == 0)
			throw new Exception("Size of sample (%) has to be greater than zero and smaller "
					+ "than or equal to 100!");
		else {
			if(checkComb)
				checkBagSizePercentAndReplacementAndNewDistrMinClassOptions(m_RMreplacement, v, m_RMnewDistrMinClass);
			m_RMbagSizePercent = v;
		}
	}

	/**
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String RMnewDistrMinClassTipText() {
		return "Determines the new value of the distribution of the minority class, if we want to change it/-1=free/-2=stratified.\n" +
				"It can be one of the following values:\n" +
				" * A value between 0 and 100 to change the portion of minority class instances in the new samples\n" +
				"   (If the dataset is multi-class, only the special value 50.0 will be accepted to balance the classes)\n" +
				" * -1 (free): Works with the instances without taking their class into account.\n" +  
				" * -2 (stratified): Maintains the original class distribution in the new samples.\n" +
				" (default: 50.0)";
	}

	/**
	 * Get the value of RMnewDistrMinClass
	 * 
	 * @return Value of RMnewDistrMinClass
	 */
	public float getRMnewDistrMinClass() {
		return m_RMnewDistrMinClass;
	}

	/**
	 * Set the value of RMnewDistrMinClass
	 * Checks the combinations of the options RMreplacement, RMbagSizePercent and RMnewDistrMinClass
	 * 
	 * @param v Value to assign to RMnewDistrMinClass
	 * @throws Exception if an option is not supported
	 */
	public void setRMnewDistrMinClass(float v) throws Exception {

		setRMnewDistrMinClass(v, true);
	}

	/**
	 * Set the value of RMnewDistrMinClass, but, optionally,
	 * checks the combinations of the options RMreplacement, RMbagSizePercent and RMnewDistrMinClass.
	 * This makes possible only checking in the last call of the method setOptions().
	 * 
	 * @param v Value to assign to RMnewDistrMinClass
	 * @param checkComb true to check
	 * @throws Exception if an option is not supported
	 */
	public void setRMnewDistrMinClass(float v, boolean checkComb) throws Exception {

		if ((v < -2) || (v == 0) || (v >= 100))
			throw new Exception("Minority class distribution has to be greater than zero and smaller " +
					"than 100 (or -1 for 'sizeOfMinClass' or -2 for 'maxSize')!");
		else {
			if (checkComb)
				checkBagSizePercentAndReplacementAndNewDistrMinClassOptions(m_RMreplacement, m_RMbagSizePercent, v);
			m_RMnewDistrMinClass = v;
		}
	}

	/**
	 * Checks the combinations of the options RMreplacement, RMbagSizePercent and RMnewDistrMinClass 
	 *
	 * @throws Exception if an option is not supported
	 */
	private void checkBagSizePercentAndReplacementAndNewDistrMinClassOptions(
			boolean replacement, int bagSizePercent, float newDistrMinClass) throws Exception{

		if((newDistrMinClass > (float)0) && (newDistrMinClass < (float)100))
			// NewDistrMinClass is a valid value to change the distribution of the sample
			if(replacement)
				throw new Exception("Using replacement isn't contemplated to change the distribution of minority class!");
		if((newDistrMinClass == (float)-1) || (newDistrMinClass == (float)-2)){
			// NewDistrMinClass = free OR stratified
			if(bagSizePercent < 0)
				throw new Exception("Size of sample (%) has to be greater than zero and smaller " +
						"than or equal to 100!");
			if((!replacement) && (bagSizePercent==100))
				System.err.println("It doesn't make sense that size of sample (%) is 100, when replacement is false!");
		}
	}

	/**
	 * Set the value of reducedErrorPruning. Turns
	 * unpruned trees off if set.
	 * (This method only accepts the default value of J48)
	 *
	 * @param v  Value to assign to reducedErrorPruning.
	 */
	@Override
	@ProgrammaticProperty
	public void setReducedErrorPruning(boolean v){
		if (v)
			throw new IllegalArgumentException("J48 option not implemented for J48Consolidated");
		super.setReducedErrorPruning(v);
	}

	/**
	 * Set the value of numFolds.
	 * (This method only accepts the default value of J48)
	 *
	 * @param v  Value to assign to numFolds.
	 */
	@Override
	@ProgrammaticProperty
	public void setNumFolds(int v) {
		if (v != 3)
			throw new IllegalArgumentException("J48 option not implemented for J48Consolidated");
		super.setNumFolds(v);
	}

	/**
	 * Set the value of binarySplits.
	 * (This method only accepts the default value of J48)
	 *
	 * @param v  Value to assign to binarySplits.
	 */
	@Override
	@ProgrammaticProperty
	public void setBinarySplits(boolean v) {
		if (v)
			throw new IllegalArgumentException("J48 option not implemented for J48Consolidated");
		super.setBinarySplits(v);
	}

	/**
	 * Returns the tip text for this property
	 * (Rewritten to indicate the true using of the seed in this class)
	 * 
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String seedTipText() {
		return "Seed for random data shuffling in the generation of samples";
	}

	/**
	 * Returns a superconcise version of the model
	 * 
	 * @return a summary of the model
	 */
	public String toSummaryString() {
		String stNumberSamplesByCoverage;
		if (m_RMnumberSamplesHowToSet == NumberSamples_BasedOnCoverage)
			stNumberSamplesByCoverage = "Number of samples based on coverage: " + measureNumberSamplesByCoverage() + "\n";
		else
			stNumberSamplesByCoverage = "";
		return super.toSummaryString() + 
				stNumberSamplesByCoverage +
				"True coverage: " + measureTrueCoverage() + "\n";
	}

	/**
	 * Returns the number of samples neccesary to achieve the indicated coverage
	 * @return number of samples based on coverage
	 */
	public double measureNumberSamplesByCoverage() {
		return m_numberSamplesByCoverage;
	}

	/**
	 * Returns the true coverage of the examples of the original sample
	 * achieved by the set of samples generated for the consolidated tree
	 * @return the true coverage achieved
	 */
	public double measureTrueCoverage() {
		return m_trueCoverage;
	}

	/**
	 * Returns an enumeration of the additional measure names
	 * produced by the J48 algorithm, plus the true coverage achieved
	 * by the set of samples generated
	 * @return an enumeration of the measure names
	 */
	public Enumeration<String> enumerateMeasures() {
		Enumeration<String> enm = super.enumerateMeasures();
		Vector<String> measures = new Vector<String>();
		while (enm.hasMoreElements())
			measures.add(enm.nextElement());
		//if (m_RMnumberSamplesHowToSet == NumberSamples_BasedOnCoverage)
		measures.add("measureNumberSamplesByCoverage");
		measures.add("measureTrueCoverage");
		return measures.elements();
	}

	/**
	 * Returns the value of the named measure
	 * @param additionalMeasureName the name of the measure to query for its value
	 * @return the value of the named measure
	 * @throws IllegalArgumentException if the named measure is not supported
	 */
	public double getMeasure(String additionalMeasureName) {
		if(Collections.list(new J48().enumerateMeasures()).contains(additionalMeasureName)) {
			return super.getMeasure(additionalMeasureName);
		} else if (additionalMeasureName.compareToIgnoreCase("measureTrueCoverage") == 0) {
			return measureTrueCoverage();
		} else if (additionalMeasureName.compareToIgnoreCase("measureNumberSamplesByCoverage") == 0) {
			return measureNumberSamplesByCoverage();
		} else
			throw new IllegalArgumentException(additionalMeasureName 
					+ " not supported (J48Consolidated)");
	}

	/**
	 * Main method for testing this class
	 *
	 * @param argv the commandline options
	 */
	public static void main(String [] argv){
		runClassifier(new J48Consolidated(), argv);
	}
}