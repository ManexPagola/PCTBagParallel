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
 *    PartiallyConsolidatedTreeBagging.java
 *    Copyright (C) 2021 ALDAPA Team (http://www.aldapa.eus)
 *    Faculty of Informatics, Donostia, 20018
 *    University of the Basque Country (UPV/EHU), Basque Country
 *
 */

package weka.classifiers.trees;


import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Enumeration;
import java.util.Vector;

import weka.classifiers.Sourcable;
import weka.classifiers.trees.j48.C45ModelSelection;
import weka.classifiers.trees.j48.ClassifierTree;
import weka.classifiers.trees.j48.ModelSelection;
import weka.classifiers.trees.j48Consolidated.C45ConsolidatedModelSelection;
import weka.classifiers.trees.j48PartiallyConsolidated.C45ModelSelectionExtended;
import weka.classifiers.trees.j48PartiallyConsolidated.C45PartiallyConsolidatedPruneableClassifierTree;
import weka.core.AdditionalMeasureProducer;
import weka.core.Drawable;
import weka.core.Instance;
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
import weka.core.matrix.DoubleVector;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

/**
<!-- globalinfo-start -->
 * Class for generating a Partially Consolidated Tree-Bagging (PCTBagging) multiple classifier.<br/>
 * Allows building a classifier between a single consolidated tree (100%), based on J48Consolidated,
 * and a bagging (0%), according to the given consolidation percent. First, a partially consolidated
 * tree is built based on a set of samples, and then, a standard J48 decision tree is developed from
 * the leaves of the tree related to each sample, as Bagging does.<p/>
 * For more information, see:<p/>
 * Igor Ibarguren and Jes&uacute;s M. P&eacute;rez and Javier Muguerza and Olatz Arbelaitz and Ainhoa Yera.  
 * "PCTBagging: From Inner Ensembles to Ensembles. A trade-off between Discriminating Capacity and Interpretability". Information Sciences (2022), Vol. 583, pp 219-238.
 * <a href="https://doi.org/10.1016/j.ins.2021.11.010" target="_blank">doi:10.1016/j.ins.2021.11.010</a>
 * <p/>
 * Also see:<br/>
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
 * &#64;article{Ibarguren2022,
 *    title = "PCTBagging: From Inner Ensembles to Ensembles. A trade-off between Discriminating Capacity and Interpretability",
 *    journal = "Information Sciences",
 *    volume = "583",
 *    pages = "219 - 238",
 *    year = "2022",
 *    doi = "10.1016/j.ins.2021.11.010",
 *    author = "Igor Ibarguren and Jes\'us M. P\'erez and Javier Muguerza and Olatz Arbelaitz and Ainhoa Yera"
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
 * Attention! Removed 'final' modifier of 'j48.distributionForInstance(...)'
 *  in order to override this function here.
 * <ul>
 *    <li>public double [] distributionForInstance(Instance instance)</li>
 * </ul>
 * *************************************************************************************<br/>
<!-- options-start -->
 * Valid options are: <p/>
 * 
 * J48 options <br/>
 * ==========
 *
 * <pre> -U
 *  Use unpruned tree.</pre>
 * 
 * <pre> -C &lt;pruning confidence&gt;
 *  Set confidence threshold for pruning.
 *  (default 0.25)</pre>
 * 
 * <pre> -M &lt;minimum number of instances&gt;
 *  Set minimum number of instances per leaf.
 *  (default 2)</pre>
 *  
 * <pre> -S
 *  Don't perform subtree raising.</pre>
 * 
 * <pre> -L
 *  Do not clean up after the tree has been built.</pre>
 * 
 * <pre> -A
 *  Laplace smoothing for predicted probabilities.</pre>
 * 
 * <pre> -Q &lt;seed&gt;
 *  Seed for random data shuffling (default 1).</pre>
 * 
 * Options to set the Resampling Method (RM) for the generation of samples
 *  to use in the consolidation process <br/>
 * =============================================================================================== 
 * <pre> -RM-C
 *  Determines that the way to set the number of samples to be generated will be based on
 *  a coverage value as a percent. In the case this option is not set, the number of samples
 *  will be determined using a fixed value. 
 *  (Set by default)</pre>
 * 
 * <pre> -RM-N &lt;number of samples&gt;
 *  Number of samples to be generated for use in the construction of the consolidated tree.
 *  It can be set as a fixed value or based on a coverage value as a percent, when -RM-C option
 *  is used, which guarantees the number of samples necessary to cover adequately the examples 
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
 *  * -2 (Max): Maximum size taking into account &lt;distribution minority class&gt;
 *  *           and using no replacement
 *  (default -2)</pre>
 *  
 * <pre> -RM-D &lt;distribution minority class&gt;
 *  Determines the new value of the distribution of the minority class, if we want to change it.
 *  It can be one of the following values:
 *  * A value between 0 and 100 to change the portion of minority class instances in the new samples
 *    (this option can only be used with binary problems (two classes datasets))
 *  * -1 (free): Works with the instances without taking into account their class  
 *  * -2 (stratified): Maintains the original class distribution in the new samples
 *  (default 50.0)</pre>
 * 
 * Options to leave partially consolidated the built consolidated tree (PCTB)
 * ============================================================================ 
 * <pre> -PCTB-C consolidation percent <br>
 * Determines the number of inner nodes to leave as consolidated as a percent 
 * according to the inner nodes of the whole consolidated tree.  
 * (Default: 20.0)<pre>
 * 
 * <pre> -PCTB-V mode <br>
 * Determines how many base trees will be shown:
 *  None, only the first ten (if they exist) or all.  
 * (Default: only the first ten)<pre>
 * 
 * <pre> -PCTB-P<br>
 * Determines whether to show the explanation aggregated measures of the all decision trees
 *  that compose the final classifier (MCS).  
 * (Default: false)<pre>
 * 
 * <pre> -PCTB-WP<br>
 * Determines whether to prune the base trees without preserving the structure of
 * the partially consolidated tree. 
 * (Default: false)<pre>
 * 
<!-- options-end -->
 *
 * @author Jesús M. Pérez (txus.perez@ehu.eus)
 * @author Ander Otsoa de Alda Alzaga (ander.otsoadealda@gmail.com)
 *  (based on J48Consolidated.java written by Jesús M. Pérez et al) 
 *  (based on J48.java written by Eibe Frank)
 * @version $Revision: 0.3 $
 */
public class J48PartiallyConsolidated
	extends J48Consolidated
	implements OptionHandler, Drawable, Matchable, Sourcable,
				WeightedInstancesHandler, Summarizable, AdditionalMeasureProducer,
				TechnicalInformationHandler {

	/** for serialization */
	private static final long serialVersionUID = 1279412716088899857L;

	/** Options related to PCTBagging algorithm
	 *  (Prefix PCTB added to the option names in order to appear together in the graphical interface)
	 ********************************************************************************/
	/** The consolidation percent. It indicates the number of inner nodes to leave as consolidated as a percent 
	 * according to the inner nodes of the whole consolidated tree. It accepts values between 0 and 100.
	 */
	protected float m_PCTBconsolidationPercent = (float)20.0;
	
	/** Options to visualize the set of base trees */
	public static final int Visualize_None = 1;
	public static final int Visualize_FirstOnes = 2;
	public static final int Visualize_All = 3;

	/** Strings related to the options to visualize the set of base trees */
	public static final Tag[] TAGS_VISUALIZE_BASE_TREES = {
		new Tag(Visualize_None, "None"),
		new Tag(Visualize_FirstOnes, "Only the first ten"),
		new Tag(Visualize_All, "All")
	};
	
	/** Visualize the base trees: None, only the first ten (if they exist) or all */
	protected int m_PCTBvisualizeBaseTrees = Visualize_FirstOnes;

	/** Whether to show the explanation aggregated measures of the all decision trees
	 *  that compose the final classifier (MCS). */
	protected boolean m_PCTBprintExplanationMeasuresBaseTrees = false;

	/** Whether to prune the base trees without preserving the structure of the partially
	 * consolidated tree. */
	protected boolean m_PCTBpruneBaseTreesWithoutPreservingConsolidatedStructure = false;

	/** Array for storing the generated base classifiers.
	 * (based on Bagging.java written by Eibe Frank eta al)
	 * */
	protected ClassifierTree[] m_Classifiers;

	/**
	 * Returns a string describing the classifier
	 * @return a description suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String globalInfo() {
		return "Class for generating a Partially Consolidated Tree-Bagging (PCTBagging) multiple classifier. "
				+ "Allows building a classifier between a single consolidated tree (100%), based on J48Consolidated, "
				+ "and a bagging (0%), according to the given consolidation percent.\n"
				+ "First, a partially consolidated tree is built based on a set of samples, and then, "
				+ "a standard J48 decision tree is developed from the leaves of the tree related to each sample, as Bagging does. "
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
		result.setValue(Field.AUTHOR, "Igor Ibarguren and Jesús M. Pérez and Javier Muguerza and Olatz Arbelaitz and Ainhoa Yera");
		result.setValue(Field.YEAR, "2022");
		result.setValue(Field.TITLE, "PCTBagging: From Inner Ensembles to Ensembles. A trade-off between Discriminating Capacity and Interpretability");
	    result.setValue(Field.JOURNAL, "Information Sciences");
	    result.setValue(Field.VOLUME, "583");
	    result.setValue(Field.PAGES, "219-238");
	    result.setValue(Field.URL, "https://doi.org/10.1016/j.ins.2021.11.010");
	    
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
	 * (based on buildClassifier() function of J48Consolidated class)
	 *
	 * @see weka.classifiers.trees.J48Consolidated#buildClassifier(weka.core.Instances)
	 */
	public void buildClassifier(Instances instances)
			throws Exception {

		// can classifier tree handle the data?
		getCapabilities().testWithFail(instances);

		// remove instances with missing class before generate samples
		instances = new Instances(instances);
		instances.deleteWithMissingClass();
		
		//Generate as many samples as the number of samples with the given instances
		//long startTime = System.nanoTime();
		Instances[] samplesVector = generateSamples(instances);
		//long endTime = System.nanoTime();
		
		//long execTime = (endTime - startTime) / 1000;
		
		//System.out.println("Laginketak " + execTime + " mikros behar izan ditu \n");
		//System.out.println("\n " + samplesVector.length + " \n" );
	    //if (m_Debug)
	    //	printSamplesVector(samplesVector);

		/** Set the model selection method to determine the consolidated decisions */
	    ModelSelection modSelection;
		// TODO Implement the option binarySplits of J48
		modSelection = new C45ConsolidatedModelSelection(m_minNumObj, instances, 
				m_useMDLcorrection, m_doNotMakeSplitPointActualValue);
		/** Set the model selection method to force the consolidated decision in each base tree*/
		C45ModelSelectionExtended baseModelToForceDecision = new C45ModelSelectionExtended(m_minNumObj, instances, 
				m_useMDLcorrection, m_doNotMakeSplitPointActualValue);
		// TODO Implement the option reducedErrorPruning of J48
		C45PartiallyConsolidatedPruneableClassifierTree localClassifier =
				new C45PartiallyConsolidatedPruneableClassifierTree(modSelection, baseModelToForceDecision,
						!m_unpruned, m_CF, m_subtreeRaising, !m_noCleanup, m_collapseTree, samplesVector.length,
						m_PCTBpruneBaseTreesWithoutPreservingConsolidatedStructure);

		localClassifier.buildClassifier(instances, samplesVector, m_PCTBconsolidationPercent);

		m_root = localClassifier;
		m_Classifiers = localClassifier.getSampleTreeVector();
		// // We could get any base tree of the vector as root and use it in the graphical interface
		// // (for example, to visualize it)
		// //m_root = localClassifier.getSampleTreeIth(0);
		
		((C45ModelSelection) modSelection).cleanup();
		((C45ModelSelection) baseModelToForceDecision).cleanup();
	}

	/**
	 * Calculates the class membership probabilities for the given test instance.
	 * (based on Bagging.java)
	 * 
	 * @param instance the instance to be classified
	 * @return predicted class probability distribution
	 * @throws Exception if distribution can't be computed successfully
	 * 
	 * @see weka.classifiers.meta.Bagging#distributionForInstance(weka.core.Instance)
	 */
	public double[] distributionForInstance(Instance instance) throws Exception {
		/** Number of Samples. */
		int numberSamples = m_Classifiers.length;
		double[] sums = new double[instance.numClasses()], newProbs;

		for (int i = 0; i < numberSamples; i++) {
			if (instance.classAttribute().isNumeric() == true) {
				sums[0] += m_Classifiers[i].classifyInstance(instance);
			} else {
				newProbs = m_Classifiers[i].distributionForInstance(instance, m_useLaplace);
				for (int j = 0; j < newProbs.length; j++)
					sums[j] += newProbs[j];
			}
		}
		if (instance.classAttribute().isNumeric() == true) {
			sums[0] /= numberSamples;
			return sums;
		} else if (Utils.eq(Utils.sum(sums), 0)) {
			return sums;
		} else {
			Utils.normalize(sums);
			return sums;
		}
	}

	/**
	 * Classifies an instance.
	 * (based on J48.java)
	 * @see weka.classifiers.trees.J48.classifyInstance(Instance)
	 * 
	 * @param instance the instance to classify
	 * @return the classification for the instance
	 * @throws Exception if instance can't be classified successfully
	 */
	@Override
	public double classifyInstance(Instance instance) throws Exception {
		double[] distribution = new double[instance.numClasses()];

		distribution = distributionForInstance(instance);
		
		double maxProb = -1;
	    double currentProb;
	    int maxIndex = 0;
	    int j;

	    for (j = 0; j < instance.numClasses(); j++) {
	      currentProb = distribution[j];
	      if (Utils.gr(currentProb, maxProb)) {
	        maxIndex = j;
	        maxProb = currentProb;
	      }
	    }

	    return maxIndex;
	}


	/**
	 * Returns an enumeration describing the available options.
	 *
	 * Valid options are: <p>
	 * 
	 * J48 options
	 * ============= 
	 *
	 * -U <br>
	 * Use unpruned tree.<p>
	 *
	 * -C confidence <br>
	 * Set confidence threshold for pruning. (Default: 0.25) <p>
	 *
	 * -M number <br>
	 * Set minimum number of instances per leaf. (Default: 2) <p>
	 *
	 * -S <br>
	 * Don't perform subtree raising. <p>
	 *
	 * -L <br>
	 * Do not clean up after the tree has been built. <p>
	 *
	 * -A <br>
	 * If set, Laplace smoothing is used for predicted probabilites. <p>
	 *
	 * -Q seed <br>
	 * Seed for random data shuffling (Default: 1) <p>
	 * 
	 * Options to set the Resampling Method (RM) for the generation of samples
	 *  to use in the consolidation process
	 * ============================================================================ 
	 * -RM-C <br>
	 * Determines that the way to set the number of samples to be generated will be based on<br>
	 * a coverage value as a percent. In the case this option is not set, the number of samples<br>
	 * will be determined using a fixed value. <br>
	 * (Set by default)<p>
	 * 
	 * -RM-N &lt;number of samples&gt;<br>
	 * Number of samples to be generated for use in the construction of the consolidated tree.<br>
	 * It can be set as a fixed value or based on a coverage value as a percent, when -RM-C option<br>
	 * is used, which guarantees the number of samples necessary to cover adequately the examples<br> 
	 * of the original sample<br>
	 * (Default 5 for a fixed value or 99% for the case based on a coverage value)<p>
	 * 
	 * -RM-R <br>
	 * Determines whether or not replacement is used when generating the samples. <br>
	 * (Default: false)<p>
	 * 
	 * -RM-B percent <br>
	 * Size of each sample(bag), as a percentage of the training set size. <br>
	 * Combined with the option &lt;distribution minority class&gt; accepts: <br>
	 *  * -1 (sizeOfMinClass): The size of the minority class <br>
	 *  * -2 (maxSize): Maximum size taking into account &lt;distribution minority class&gt;
	 *              and using no replacement <br>
	 * (Default: -2(maxSize)) <p>
	 * 
	 * -RM-D distribution minority class <br>
	 * Determines the new value of the distribution of the minority class, if we want to change it. <br>
	 * It can be one of the following values: <br>
	 *  * A value between 0 and 100 to change the portion of minority class instances in the new samples <br>
	 *    (If the dataset is multi-class, only the special value 50.0 will be accepted to balance the classes)
	 *  * -1 (free): Works with the instances without taking into account their class <br>
	 *  * -2 (stratified): Maintains the original class distribution in the new samples <br>
	 * (Default: -1(free)) <p>
	 * 
	 * Options to leave partially consolidated the built consolidated tree (PCTB)
	 * ============================================================================ 
	 * -PCTB-C consolidation percent <br>
	 * Determines the number of inner nodes to leave as consolidated as a percent 
	 * according to the inner nodes of the whole consolidated tree.  
	 * (Default: 20.0)<p>
	 * 
	 * <pre> -PCTB-V mode <br>
	 * Determines how many base trees will be shown:
	 *  None, only the first ten (if they exist) or all.  
	 * (Default: only the first ten)<pre>
	 * 
	 * <pre> -PCTB-P<br>
	 * Determines whether to show the explanation aggregated measures of the all decision trees
	 *  that compose the final classifier (MCS).  
	 * (Default: false)<pre>
	 * 
	 * <pre> -PCTB-WP<br>
	 * Determines whether to prune the base trees without preserving the structure of
	 * the partially consolidated tree. 
	 * (Default: false)<pre>
	 * 
	 * @return an enumeration of all the available options.
	 */
	public Enumeration<Option> listOptions() {

		Vector<Option> newVector = new Vector<Option>();

		// J48 and J48Consolidated options
		// ===============================
	    Enumeration<Option> en;
	    en = super.listOptions();
	    while (en.hasMoreElements())
	    	newVector.addElement((Option) en.nextElement());

		// Options to leave partially consolidated the built consolidated tree (PCTB)
		// =========================================================================
		newVector.
		addElement(new Option(
				"\tDetermines the number of inner nodes to leave as consolidated as a percent\n" +
				"\taccording to the inner nodes of the whole consolidated tree.\n" +
				"\t(default 20.0)",
				"PCTB-C", 1, "-PCTB-C <consolidation percent>"));

		newVector.
		addElement(new Option(
				"\tDetermines how many base trees will be shown:\n" +
				"\tNone, only the first ten (if they exist) or all.\n" +
				"\t(default: only the first ten)",
				"PCTB-V", 1, "-PCTB-V <mode>"));

	    newVector.
	    addElement(new Option(
	            "\tDetermines whether to show the explanation aggregated measures of the all base trees" +
	            "\tthat compose the final classifier (MCS).\n" + 
	            "\t(default false)",
	            "PCTB-P", 0, "-PCTB-P"));

	    newVector.
	    addElement(new Option(
	            "\tDetermines whether to prune the base trees without preserving the structure " +
	            "\tthe partially consolidated tree.\n" + 
	            "\t(default false)",
	            "PCTB-WP", 0, "-PCTB-WP"));

	    return newVector.elements();
	}
	/**
	 * Parses a given list of options.
	 * 
   <!-- options-start -->
	 * Valid options are: <p/>
	 * 
	 * Options to set the Resampling Method (RM) for the generation of samples
	 *  to use in the consolidation process
	 * ============================================================================ 
	 * <pre> -RM-C
	 *  Determines that the way to set the number of samples to be generated will be based on
	 *  a coverage value as a percent. In the case this option is not set, the number of samples
	 *  will be determined using a fixed value. 
	 *  (Set by default)</pre>
	 * 
	 * <pre> -RM-N &lt;number of samples&gt;
	 *  Number of samples to be generated for use in the construction of the consolidated tree.
	 *  It can be set as a fixed value or based on a coverage value as a percent, when -RM-C option
	 *  is used, which guarantees the number of samples necessary to cover adequately the examples 
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
	 *  * -2 (maxSize): Maximum size taking into account &lt;distribution minority class&gt;
	 *  *           and using no replacement
	 *  (default -2(maxSize))</pre>
	 * 
	 * <pre> -RM-D &lt;distribution minority class&gt;
	 *  Determines the new value of the distribution of the minority class, if we want to change it.
	 *  It can be one of the following values:
	 *  * A value between 0 and 100 to change the portion of minority class instances in the new samples
	 *    (If the dataset is multi-class, only the special value 50.0 will be accepted to balance the classes)
	 *  * -1 (free): Works with the instances without taking into account their class  
	 *  * -2 (stratified): Maintains the original class distribution in the new samples
	 *  (default 50.0)</pre>
	 * 
	 * Options to leave partially consolidated the built consolidated tree (PCTB)
	 * ============================================================================ 
	 * <pre> -PCTB-C consolidation percent
	 * Determines the number of inner nodes to leave as consolidated as a percent 
	 * according to the inner nodes of the whole consolidated tree.  
	 * (Default: 20.0)</pre>
	 * 
	 * <pre> -PCTB-V mode <br>
	 * Determines how many base trees will be shown:
	 *  None, only the first ten (if they exist) or all.  
	 * (Default: only the first ten)<pre>
	 * 
	 * <pre> -PCTB-P<br>
	 * Determines whether to show the explanation aggregated measures of the all decision trees
	 *  that compose the final classifier (MCS).  
	 * (Default: false)<pre>
	 * 
	 * <pre> -PCTB-WP<br>
	 * Determines whether to prune the base trees without preserving the structure of
	 * the partially consolidated tree. 
	 * (Default: false)<pre>
	 * 
   <!-- options-end -->
	 *
	 * @param options the list of options as an array of strings
	 * @throws Exception if an option is not supported
	 */
	/** Whether to prune the base trees without preserving the structure of the partially
	 * consolidated tree. */

	public void setOptions(String[] options) throws Exception {
	    
		// Options to leave partially consolidated the built consolidated tree (PCTB)
		// =========================================================================
		String PCTBconsolidationPercentString = Utils.getOption("PCTB-C", options);
		if (PCTBconsolidationPercentString.length() != 0)
			setPCTBconsolidationPercent(Float.valueOf(PCTBconsolidationPercentString));
		else
			setPCTBconsolidationPercent((float)20.0);
		String PCTBvisualizeBaseTreesString = Utils.getOption("PCTB-V", options);
		if (PCTBvisualizeBaseTreesString.length() != 0)
			setPCTBvisualizeBaseTrees(new SelectedTag(Integer.parseInt(PCTBvisualizeBaseTreesString), TAGS_VISUALIZE_BASE_TREES));
		else
			setPCTBvisualizeBaseTrees(new SelectedTag(Visualize_FirstOnes, TAGS_VISUALIZE_BASE_TREES)); // default: only the first ten
		
	    setPCTBprintExplanationMeasuresBaseTrees(Utils.getFlag("PCTB-P", options));

	    setPCTBpruneBaseTreesWithoutPreservingConsolidatedStructure(Utils.getFlag("PCTB-WP", options));

	    // J48 and J48Consolidated options
		// ===============================
	    super.setOptions(options);
	}

	/**
	 * Gets the current settings of the Classifier.
	 *
	 * @return an array of strings suitable for passing to setOptions
	 */
	public String [] getOptions() {

		Vector<String> result = new Vector<String>();

		// J48 and J48Consolidated options
		// ===============================
		String[] options = super.getOptions();
		for (int i = 0; i < options.length; i++)
			result.add(options[i]);

		// Options to leave partially consolidated the built consolidated tree (PCTB)
		// =========================================================================
		result.add("-PCTB-C");
		result.add("" + m_PCTBconsolidationPercent);

		result.add("-PCTB-V");
		result.add("" + m_PCTBvisualizeBaseTrees);

		if (m_PCTBprintExplanationMeasuresBaseTrees)
			result.add("-PCTB-P");

		if (m_PCTBpruneBaseTreesWithoutPreservingConsolidatedStructure)
			result.add("-PCTB-WP");

		return (String[]) result.toArray(new String[result.size()]);	  
	}

	/**
	 * Returns a description of the classifier.
	 * 
	 * @return a description of the classifier
	 */
	public String toString() {
		String st;
		st = "J48PartiallyConsolidated-Bagging classifier\n";
		// Add a separator
		char[] ch_line = new char[st.length()];
		for (int i = 0; i < ch_line.length; i++)
			ch_line[i] = '-';
		String line = String.valueOf(ch_line);
		line += "\n";
		st += "Consolidation percent = " + Utils.doubleToString(m_PCTBconsolidationPercent,2) + "%";
		int numberNodesConso = ((C45PartiallyConsolidatedPruneableClassifierTree)m_root).getNumInternalNodesConso();
		if (numberNodesConso >= 0)
			st += " => Internal nodes: " + numberNodesConso;
		st += "\n";
		if (m_PCTBpruneBaseTreesWithoutPreservingConsolidatedStructure)
			st += "·Without preserving the structure of the partially consolidated tree in the base trees\n";
		else
			st += "·Preserving the structure of the partially consolidated tree in the base trees\n";
		st += line;
		st += super.toString();
		st += toStringVisualizeBaseTrees(line);
		st += toStringPrintExplanationMeasuresMCS();
		return st;
	}

	/**
	 * Returns the base trees as text
	 * 
	 * @return the base trees as text
	 */
	public String toStringVisualizeBaseTrees(String line) {
		String st = "";
		if (m_PCTBvisualizeBaseTrees > Visualize_None) {
			/** Base tree vector */
			if (m_Classifiers != null){
				int maxBaseTrees;
				st += line;
				if ((m_PCTBvisualizeBaseTrees == Visualize_All) || (m_Classifiers.length <= 10)) {
					maxBaseTrees = m_Classifiers.length;
					st += "Set of " + m_Classifiers.length + " base " + (m_unpruned? "unpruned " : "") + "trees:\n";
				}
				else {
					maxBaseTrees = 10;
					st += "Set of " + m_Classifiers.length + " base " + (m_unpruned? "unpruned " : "") + "trees (*Only the first ten!):\n";
				}
				for (int iSample = 0; iSample < maxBaseTrees; iSample++){
					st += line;
					st += iSample + "-th base tree:\n";
					st += m_Classifiers[iSample].toString();
				}
			}
		}
		return st;
	}

	/**
	 * Returns a description of the explanation measures of the all decision trees
	 *  that compose the final classifier (MCS)
	 * 
	 * @return a description of the explanation measures of all DTs
	 */
	public String toStringPrintExplanationMeasuresMCS() {
		String st = "";
		if (m_PCTBprintExplanationMeasuresBaseTrees) {
			st += "\n--- Complexity/Explanation aggregated measures  ---"
				+ "\n---  of the whole multiple classifier           ---"
				+ "\n---------------------------------------------------\n";
			
			String[] stMetaOperations = new String[]{"Avg", "Min", "Max", "Sum", "Mdn", "Dev"};
			ArrayList<String> metaOperations = new ArrayList<>(Arrays.asList(stMetaOperations));
			String[] stTreeMeasures = new String[]{"NumLeaves", /*"NumRules",*/ "NumInnerNodes", "ExplanationLength", "WeightedExplanationLength"};
			ArrayList<String> treeMeasures = new ArrayList<>(Arrays.asList(stTreeMeasures));

			for(String ms: treeMeasures) {
				st += ms + ": ";
				for(String op: metaOperations) {
					String mtms = "measure" + op + ms;
					st += op + " = " + Utils.roundDouble(getMeasure(mtms),2) + " ";
				}
				st += "\n";
			}
		}
		return st;
	}

	/**
	 * Returns the meta-classifier as Java source code.
	 * (based on toSource() function of AdaBoostM1 class)
	 *
	 * @see weka.classifiers.meta.AdaBoostM1.toSource(String)
	 *
	 * @param className the classname of the generated code
	 * @return the tree as Java source code
	 * @throws Exception if something goes wrong
	 */
	public String toSource(String className) throws Exception {
		if (m_Classifiers == null) {
			throw new Exception("No model built yet");
		}
		Instances data = m_root.getTrainingData();
		int numClasses = data.numClasses();

		/** Number of Samples. */
		int numberSamples = m_Classifiers.length;
		StringBuffer text = new StringBuffer("class ");
		text.append(className).append(" {\n\n");

		text.append("  public static double classify(Object[] i) throws Exception {\n");

		if (numberSamples == 1) {
			text.append("    return " + className + "_0.classify(i);\n");
		} else {
			text.append("    double [] sums = new double [" + numClasses + "];\n");
			for (int i = 0; i < numberSamples; i++) {
				text.append("    sums[(int) " + className + '_' + i
						+ ".classify(i)] += (double)1 ;\n");
			}
			text.append("    double maxV = sums[0];\n" + "    int maxI = 0;\n"
					+ "    for (int j = 1; j < " + numClasses + "; j++) {\n"
					+ "      if (sums[j] > maxV) { maxV = sums[j]; maxI = j; }\n"
					+ "    }\n    return (double) maxI;\n");
		}
		text.append("  }\n}\n");

		for (int i = 0; i < m_Classifiers.length; i++) {
			/**
			 * Based on toSource() function of J48 class
			 *
			 * @see weka.classifiers.trees.J48.toSource(String) 
			 */
			StringBuffer[] sourceTree = ((ClassifierTree) m_Classifiers[i]).toSource(className + '_' + i);
			String sourceClas = "class " + className + '_' + i + " {\n\n"
				      + "  public static double classify(Object[] i)\n"
				      + "    throws Exception {\n\n" + "    double p = Double.NaN;\n"
				      + sourceTree[0] // Assignment code
				      + "    return p;\n" + "  }\n" + sourceTree[1] // Support code
				      + "}\n";
			text.append(sourceClas);
		}
		return text.toString();
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
		newVector.addAll(Collections.list(super.enumerateMeasures()));
		String[] stMetaOperations = new String[]{"Avg", "Min", "Max", "Sum", "Mdn", "Dev"};
		ArrayList<String> metaOperations = new ArrayList<>(Arrays.asList(stMetaOperations));
		String[] stTreeMeasures = new String[]{"NumLeaves", "NumRules", "NumInnerNodes", "ExplanationLength", "WeightedExplanationLength"};
		ArrayList<String> treeMeasures = new ArrayList<>(Arrays.asList(stTreeMeasures));

		for(String op: metaOperations)
			for(String ms: treeMeasures)
				newVector.addElement("measure" + op + ms);

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
	 *
	 * @param additionalMeasureName the name of the measure to query for its value
	 * @return the value of the named measure
	 * @throws IllegalArgumentException if the named measure is not supported
	 */
	@Override
	public double getMeasure(String additionalMeasureName) {
		if(Collections.list(new J48Consolidated().enumerateMeasures()).contains(additionalMeasureName)) {
			return super.getMeasure(additionalMeasureName);
		} else {
			String[] stMetaOperations = new String[]{"Avg", "Min", "Max", "Sum", "Mdn", "Dev"};
			ArrayList<String> metaOperations = new ArrayList<>(Arrays.asList(stMetaOperations));
			String[] stTreeMeasures = new String[]{"NumLeaves", "NumRules", "NumInnerNodes", "ExplanationLength", "WeightedExplanationLength"};
			ArrayList<String> treeMeasures = new ArrayList<>(Arrays.asList(stTreeMeasures));
	
			// If it is an aggregate measure associated to the meta-classifier...
			for(String op: metaOperations)
				for(String ms: treeMeasures) {
					String mtms = "measure" + op + ms;
					if (additionalMeasureName.equalsIgnoreCase(mtms)) {
						String sms = "measure" + ms;
						int iop = metaOperations.indexOf(op);
						return getMetaAggregateMeasure(sms, iop);
					}
				}
			throw new IllegalArgumentException(additionalMeasureName 
					+ " not supported (J48PartiallyConsolidated)");
		}
	}

	/**
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String PCTBconsolidationPercentTipText() {
		return "Consolidation percent for use after the consolidation process";
	}

	/**
	 * Get the value of ConsolidationPercent.
	 *
	 * @return Value of ConsolidationPercent.
	 */
	public float getPCTBconsolidationPercent() {

		return m_PCTBconsolidationPercent;
	}

	/**
	 * Set the value of ConsolidationPercetnt.
	 *
	 * @param v Value to assign to ConsolidationPercent.
	 * @throws Exception if an option is not supported
	 */
	public void setPCTBconsolidationPercent(float v) throws Exception {
		if ((v < 0) || (v > 100))
			throw new Exception("The consolidation percent (%) has to be a value greater than or equal to zero and smaller " +
					"than or equal to 100!");
		m_PCTBconsolidationPercent = v;
	}

	/**
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String PCTBvisualizeBaseTreesTipText() {
		return "Mode to visualize the set of base trees: None, only the first ten or all";
	}

	/**
	 * Get the value of visualizeBaseTrees.
	 *
	 * @return Value of visualizeBaseTrees.
	 */
	public SelectedTag getPCTBvisualizeBaseTrees() {
	    return new SelectedTag(m_PCTBvisualizeBaseTrees,
	    		TAGS_VISUALIZE_BASE_TREES);
	}

	/**
	 * Set the value of visualizeBaseTrees.
	 *
	 * @param mode Value to assign to visualizeBaseTrees.
	 * @throws Exception if an option is not supported
	 */
	public void setPCTBvisualizeBaseTrees(SelectedTag mode) throws Exception {
	  	if (mode.getTags() == TAGS_VISUALIZE_BASE_TREES) 
	  	{
	  		int newMode = mode.getSelectedTag().getID();
	  		
		    if (newMode == Visualize_None || newMode == Visualize_FirstOnes || newMode == Visualize_All)
		    	m_PCTBvisualizeBaseTrees = newMode;
		    else 
		    	throw new IllegalArgumentException("Wrong selection type, value should be: "
		                                           + "between 1 and 3");
		 }
	}

	/**
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String PCTBprintExplanationMeasuresBaseTreesTipText() {
		return "Whether to show the explanation aggregated measures of all base trees.";
	}
	
	/**
	 * Set whether to show the explanation aggregated measures of all base trees. 
	 *
	 * @param printExplanationMeasures whether to show the explanation measures of all DTs.
	 */
	public void setPCTBprintExplanationMeasuresBaseTrees(boolean printExplanationMeasures) {

		m_PCTBprintExplanationMeasuresBaseTrees = printExplanationMeasures;
	}

	/**
	 * Get whether to show the explanation aggregated measures of all base trees.
	 *
	 * @return whether to to show the explanation measures
	 */
	public boolean getPCTBprintExplanationMeasuresBaseTrees() {

		return m_PCTBprintExplanationMeasuresBaseTrees;
	}

	/**
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String PCTBpruneBaseTreesWithoutPreservingConsolidatedStructureTipText() {
		return "Whether to prune the base trees without preserving the structure of the partially consolidated tree.";
	}

	/**
	 * Set whether to prune the base trees without preserving the structure of the partially
	 * consolidated tree. 
	 *
	 * @param notPreserving whether to prune without preserving the consolidated structure.
	 */
	public void setPCTBpruneBaseTreesWithoutPreservingConsolidatedStructure(boolean notPreserving) {

		m_PCTBpruneBaseTreesWithoutPreservingConsolidatedStructure = notPreserving;
	}

	/**
	 * Get whether to prune the base trees without preserving the structure of the partially
	 * consolidated tree.
	 *
	 * @return whether to prune without preserving the consolidated structure
	 */
	public boolean getPCTBpruneBaseTreesWithoutPreservingConsolidatedStructure() {

		return m_PCTBpruneBaseTreesWithoutPreservingConsolidatedStructure;
	}

}