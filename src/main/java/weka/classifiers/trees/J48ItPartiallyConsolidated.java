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

import java.util.Collections;
import java.util.Enumeration;
import java.util.Vector;

import weka.classifiers.Sourcable;
import weka.classifiers.trees.j48.C45ModelSelection;
import weka.classifiers.trees.j48.ModelSelection;
import weka.classifiers.trees.j48Consolidated.C45ConsolidatedModelSelection;
import weka.classifiers.trees.j48ItPartiallyConsolidated.C45ItPartiallyConsolidatedPruneableClassifierTree;
import weka.classifiers.trees.j48PartiallyConsolidated.C45ModelSelectionExtended;
import weka.classifiers.trees.j48PartiallyConsolidated.C45PartiallyConsolidatedPruneableClassifierTree;
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
<!-- options-end -->
 *
 * @author Jesús M. Pérez (txus.perez@ehu.eus)
 * @author Ander Otsoa de Alda Alzaga (ander.otsoadealda@gmail.com)
 *  (based on J48Consolidated.java written by Jesús M. Pérez et al) 
 *  (based on J48.java written by Eibe Frank)
 * @version $Revision: 0.3 $
 */
public class J48ItPartiallyConsolidated
	extends J48PartiallyConsolidated
	implements OptionHandler, Drawable, Matchable, Sourcable,
				WeightedInstancesHandler, Summarizable, AdditionalMeasureProducer,
				TechnicalInformationHandler {

	/** for serialization */
	private static final long serialVersionUID = 1279412716088899857L;

	/** Options related to PCTBagging algorithm
	 *  (Prefix PCTB added to the option names in order to appear together in the graphical interface)
	 ********************************************************************************/

	
	/** Ways to set the priority criteria option */
	public static final int Original = 0;
	public static final int Levelbylevel = 1;
	public static final int Preorder = 2;
	public static final int Size = 3;
	public static final int Gainratio = 4;
	public static final int Gainratio_normalized = 5;

	
	/** Strings related to the ways to set the priority criteria option */
	public static final Tag[] TAGS_WAYS_TO_SET_PRIORITY_CRITERIA = {
			new Tag(Original, "Original (recursive)"),
			new Tag(Levelbylevel, "Level by level"),
			new Tag(Preorder, "Node by node - Preorder"),
			new Tag(Size, "Node by node - Size"),
			new Tag(Gainratio, "Node by node - Gainratio"),
			new Tag(Gainratio_normalized, "Node by node - Normalized gainratio"),

	};
	
	private int m_ITPCTpriorityCriteria = Original;
	
	/** Ways to set the consolidationNumber option */
	public static final int ConsolidationNumber_Value = 1;
	public static final int ConsolidationNumber_Percentage = 2;

	/** Strings related to the ways to set the numberSamples option */
	public static final Tag[] TAGS_WAYS_TO_SET_CONSOLIDATION_NUMBER = {
			new Tag(ConsolidationNumber_Value, "using a numeric value"),
			new Tag(ConsolidationNumber_Percentage, "based on a pertentage value (%)"),
	};
		
	/** Selected way to set the number of samples to be generated; or using a fixed value;
	 *   or based on a coverage value as a percentage (by default). */
	private int m_ITPCTconsolidationPercentHowToSet = ConsolidationNumber_Percentage;
	
	/** Unpruned partial Consolidated Tree (CT)? */
	private boolean m_ITPCTunprunedCT = true;
	
	/** Collapse partial Consolidated Tree (CT)? */
	private boolean m_ITPCTcollapseCT = false;
	
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
		Instances[] samplesVector = generateSamples(instances);
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
		C45ItPartiallyConsolidatedPruneableClassifierTree localClassifier =
				new C45ItPartiallyConsolidatedPruneableClassifierTree(modSelection, baseModelToForceDecision,
						!m_unpruned, m_CF, m_subtreeRaising, !m_noCleanup, m_collapseTree, samplesVector.length,
						m_PCTBpruneBaseTreesWithoutPreservingConsolidatedStructure,
						m_ITPCTpriorityCriteria, !m_ITPCTunprunedCT, m_ITPCTcollapseCT);

		localClassifier.buildClassifier(instances, samplesVector, m_PCTBconsolidationPercent, m_ITPCTconsolidationPercentHowToSet);

		m_root = localClassifier;
		m_Classifiers = localClassifier.getSampleTreeVector();
		// // We could get any base tree of the vector as root and use it in the graphical interface
		// // (for example, to visualize it)
		// //m_root = localClassifier.getSampleTreeIth(0);
		
		((C45ModelSelection) modSelection).cleanup();
		((C45ModelSelection) baseModelToForceDecision).cleanup();
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
	 * @return an enumeration of all the available options.
	 */
	
	/**
	 * Returns an enumeration describing the available options.
	 * 
	 * Valid options are:
	 * <p>
	 * 
	 * J48 options<br/>
	 * =============<br/>
	 *
	 * Options to build the tree partially (the most important change is that it is
	 * run iteratively (IT) instead of recursively).
	 * ============================================================================
	 * -ITPCT-MC <br>
	 * Build the tree with a maximum number of levels or nodes.
	 * <p>
	 * 
	 * -ITPCT-U <br>
	 * Use unpruned partial Consolidated Tree (CT).
	 * <p>
	 * 
	 * -ITPCT-C <br>
	 * Collapse the partial Consolidated Tree (CT).
	 * <p>
	 * 
	 * @return an enumeration of all the available options.
	 */
	@Override
	public Enumeration<Option> listOptions() {
		Vector<Option> newVector = new Vector<Option>(1);

		newVector.addElement(new Option("\tBuild the tree with a maximum number of levels or nodes.", "ITPCT-MC", 0, "-ITPCT-MC"));
		newVector.addElement(new Option("\tBuild the tree as it was originally built.", "ITPCT-PO", 0, "-ITPCT-PO"));
		newVector.addElement(new Option("\tBuild the tree level by level.", "ITPCT-PL", 0, "-ITPCT-PL"));
		newVector.addElement(new Option("\tBuild the tree in preorder.", "ITPCT-PP", 0, "-ITPCT-PP"));
		newVector.addElement(new Option("\tBuild the tree ordered by size.", "ITPCT-PS", 0, "-ITPCT-PS"));
		newVector.addElement(new Option("\tBuild the tree ordered by gainratio.", "ITPCT-PG", 0, "-ITPCT-PG"));
		newVector.addElement(new Option("\tBuild the tree ordered by normalized gainratio.", "ITPCT-PGN", 0, "-ITPCT-PGN"));
		newVector.addElement(new Option("\tSet the number of nodes or levels to be generated based on a value\\n\" +\n"
				+ "				\"\\tas a percentage (by default)", "ITPCT-P", 0, "-ITPCT-P"));
		newVector.addElement(new Option("\tSet the number of nodes or levels to be generated based on a numeric value", "ITPCT-V", 0, "-ITPCT-V"));
		
	    newVector.addElement(new Option("\tUse unpruned partial Consolidated Tree.", "ITPCT-U", 0, "-ITPCT-U"));
	    newVector.addElement(new Option("\tCollpase the partial Consolidated Tree.", "ITPCT-C", 0, "-ITPCT-C"));

	    newVector.addAll(Collections.list(super.listOptions()));
		return newVector.elements();
	}
	
	/**
	 * Returns an enumeration describing the available options.
	 * 
	 * Valid options are:
	 * <p>
	 * 
	 * J48 options<br/>
	 * =============<br/>
	 *
	 * Options to build the tree partially (the most important change is that it is
	 * run iteratively (IT) instead of recursively).
	 * ============================================================================
	 * 
	 * -ITPCT-P <br>
	 * Build the tree ordered by a criteria.
	 * <p>
	 * 
	 * -ITPCT-U <br>
	 * Use unpruned partial Consolidated Tree (CT).
	 * <p>
	 * 
	 * -ITPCT-C <br>
	 * Collapse the partial Consolidated Tree (CT).
	 * <p>
	 * 
	 * @return an enumeration of all the available options.
	 */
	@Override
	public void setOptions(String[] options) throws Exception {

		if (Utils.getFlag("ITPCT-P", options))
			setITPCTconsolidationPercentHowToSet(new SelectedTag(ConsolidationNumber_Percentage, TAGS_WAYS_TO_SET_CONSOLIDATION_NUMBER));
		else if (Utils.getFlag("ITPCT-V", options))
			setITPCTconsolidationPercentHowToSet(new SelectedTag(ConsolidationNumber_Value, TAGS_WAYS_TO_SET_CONSOLIDATION_NUMBER));
				
		
		if (Utils.getFlag("ITPCT-PO", options))
			setITPCTpriorityCriteria(new SelectedTag(Original, TAGS_WAYS_TO_SET_PRIORITY_CRITERIA));
		else if (Utils.getFlag("ITPCT-PL", options))
			setITPCTpriorityCriteria(new SelectedTag(Levelbylevel, TAGS_WAYS_TO_SET_PRIORITY_CRITERIA));
		else if (Utils.getFlag("ITPCT-PP", options))
			setITPCTpriorityCriteria(new SelectedTag(Preorder, TAGS_WAYS_TO_SET_PRIORITY_CRITERIA));
		else if (Utils.getFlag("ITPCT-PS", options))
			setITPCTpriorityCriteria(new SelectedTag(Size, TAGS_WAYS_TO_SET_PRIORITY_CRITERIA));
		else if (Utils.getFlag("ITPCT-PG", options))
			setITPCTpriorityCriteria(new SelectedTag(Gainratio, TAGS_WAYS_TO_SET_PRIORITY_CRITERIA));
		else if (Utils.getFlag("ITPCT-PGN", options))
			setITPCTpriorityCriteria(new SelectedTag(Gainratio_normalized, TAGS_WAYS_TO_SET_PRIORITY_CRITERIA));
		
		m_ITPCTunprunedCT = Utils.getFlag("ITPCT-U", options);
		m_ITPCTcollapseCT = Utils.getFlag("ITPCT-C", options);

	    super.setOptions(options);
	}

	/**
	 * Gets the current settings of the Classifier.
	 * 
	 * @return an array of strings suitable for passing to setOptions
	 */
	@Override
	public String[] getOptions() {

		Vector<String> options = new Vector<String>();
		Collections.addAll(options, super.getOptions());
	    
		switch(m_ITPCTpriorityCriteria) {
			case Original:
				options.add("-ITPCT-PO");break;
			case Levelbylevel:
				options.add("-ITPCT-PL");break;
			case Preorder:
				options.add("-ITPCT-PP");break;
			case Size:
				options.add("-ITPCT-PS");break;
			case Gainratio:
				options.add("-ITPCT-PG");break;
			case Gainratio_normalized:
				options.add("-ITPCT-PGN");break;
		}
		if (m_ITPCTconsolidationPercentHowToSet == ConsolidationNumber_Value) options.add("-ITPCT-V");
		else options.add("-ITPCT-P");

		if (m_ITPCTunprunedCT) {
			options.add("-ITPCT-U");
		}

		if (m_ITPCTcollapseCT) {
			options.add("-ITPCT-C");
		}

	    return options.toArray(new String[0]);
	}

	/**
	 * Returns a description of the classifier.
	 * 
	 * @return a description of the classifier
	 */
	public String toString() {
		String st;
		st = "J48ItPartiallyConsolidated-Bagging classifier\n";
		// Add a separator
		char[] ch_line = new char[st.length()];
		for (int i = 0; i < ch_line.length; i++)
			ch_line[i] = '-';
		String line = String.valueOf(ch_line);
		line += "\n";
		st += "Priority criteria to grow the partial consolidated tree: ";
		switch(m_ITPCTpriorityCriteria) {
			case Original:
				st += "Original (recursive)";break;
			case Levelbylevel:
				st += "Level by level";break;
			case Preorder:
				st += "Node by node - Preorder";break;
			case Size:
				st += "Node by node - Size";break;
			case Gainratio:
				st += "Node by node - Gainratio";break;
			case Gainratio_normalized:
				st += "Node by node - Gainratio weighted by Size";break;
		}
		st += "\n";
		if (m_ITPCTconsolidationPercentHowToSet == ConsolidationNumber_Percentage) {
			if (m_ITPCTpriorityCriteria == Levelbylevel) {
				st += "Consolidation percent (in terms of number of levels of the tree) = " + Utils.doubleToString(m_PCTBconsolidationPercent,2) + "%";
				if (m_root != null) {
					int numberNodesConso = ((C45PartiallyConsolidatedPruneableClassifierTree)m_root).getNumInternalNodesConso();
					if (numberNodesConso >= 0)
						st += " => Number of levels to grow: " + numberNodesConso;
				}
			} else {
				st += "Consolidation percent = " + Utils.doubleToString(m_PCTBconsolidationPercent,2) + "%";
				if (m_root != null) {
					int numberNodesConso = ((C45PartiallyConsolidatedPruneableClassifierTree)m_root).getNumInternalNodesConso();
					if (numberNodesConso >= 0)
						st += " => Internal nodes to grow: " + numberNodesConso;
				}
			}
			st += "\n";
		} else { // ConsolidationNumber_Value
			if (m_ITPCTpriorityCriteria == Levelbylevel)
				st += "Number of levels of the partial consolidated tree to grow = " + Utils.doubleToString(m_PCTBconsolidationPercent,0) + "\n";
			else
				st += "Number of inner nodes of the partial consolidated tree to grow = " + Utils.doubleToString(m_PCTBconsolidationPercent,0) + "\n";
		}
		if (m_PCTBpruneBaseTreesWithoutPreservingConsolidatedStructure)
			st += "·Without preserving the structure of the partially consolidated tree in the base trees\n";
		else
			st += "·Preserving the structure of the partially consolidated tree in the base trees\n";
		st += line;
		//st += super.toString();
		if (m_root == null)
			st += "No classifier built";
		else
			st += "J48Consolidated " + (m_ITPCTunprunedCT? "unpruned " : "") + (m_ITPCTcollapseCT? "(collapsed) " : "") + "tree\n" + 
				toStringResamplingMethod() + m_root.toString();
		st += toStringVisualizeBaseTrees(line);
		st += toStringPrintExplanationMeasuresMCS();
		return st;
	}

	/**
	 * Returns the time taken to build the whole Consolidated Tree (CT),
	 * including pruning/collapsing, if required.
	 * 
	 * @return elapsed time
	 */
	public double measureElapsedTimeTrainingWholeCT() {
		return ((C45ItPartiallyConsolidatedPruneableClassifierTree)m_root).getElapsedTimeTrainingWholeCT();
	}
	
	/**
	 * Returns the time taken to build the partial Consolidated Tree (CT),
	 * including pruning/collapsing, if required.
	 * 
	 * @return elapsed time
	 */
	public double measureElapsedTimeTrainingPartialCT() {
		return ((C45ItPartiallyConsolidatedPruneableClassifierTree)m_root).getElapsedTimeTrainingPartialCT();
	}

	/**
	 * Returns the time taken to build all base trees that compose the final multiple classifier (Bagging),
	 * including pruning/collapsing, if required.
	 * 
	 * @return elapsed time
	 */
	public double measureElapsedTimeTrainingAssocBagging() {
		return ((C45ItPartiallyConsolidatedPruneableClassifierTree)m_root).getElapsedTimeTrainingAssocBagging();
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
		newVector.addElement("measureElapsedTimeTrainingWholeCT");
		newVector.addElement("measureElapsedTimeTrainingPartialCT");
		newVector.addElement("measureElapsedTimeTrainingAssocBagging");
		return newVector.elements();
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
		if(Collections.list(new J48PartiallyConsolidated().enumerateMeasures()).contains(additionalMeasureName)) {
			return super.getMeasure(additionalMeasureName);
		} else  if (additionalMeasureName.compareToIgnoreCase("measureElapsedTimeTrainingWholeCT") == 0) {
			return measureElapsedTimeTrainingWholeCT();
		} else if (additionalMeasureName.compareToIgnoreCase("measureElapsedTimeTrainingPartialCT") == 0) {
			return measureElapsedTimeTrainingPartialCT();
		} else if (additionalMeasureName.compareToIgnoreCase("measureElapsedTimeTrainingAssocBagging") == 0) {
			return measureElapsedTimeTrainingAssocBagging();
		} else
			throw new IllegalArgumentException(additionalMeasureName 
					+ " not supported (J48ItPartiallyConsolidated)");
	}
	/**
	 * Returns the tip text for this property
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String ITPCTpriorityCriteriaTipText() {
		return "Build the tree ordered by a criteria: Original (recursive), LevelByLevel, Preorder, Size, Gainratio, Normalized Gainratio";
	}

	/**
	 * Get the value of ITPCTpriorityCriteria.
	 * 
	 * @return Value of ITPCTpriorityCriteria.
	 */
	public SelectedTag getITPCTpriorityCriteria() {
		return new SelectedTag(m_ITPCTpriorityCriteria,
				TAGS_WAYS_TO_SET_PRIORITY_CRITERIA);
	}

	/**
	 * Set the value of ITPCTpriorityCriteria.
	 * 
	 * @param v Value to assign to ITPCTpriorityCriteria.
	 */

	public void setITPCTpriorityCriteria(SelectedTag newPriorityCriteria) throws Exception {
		if (newPriorityCriteria.getTags() == TAGS_WAYS_TO_SET_PRIORITY_CRITERIA) 
		{
			int newPriority = newPriorityCriteria.getSelectedTag().getID();

			if (newPriority == Original || newPriority == Levelbylevel || newPriority == Preorder || newPriority == Size || newPriority == Gainratio || newPriority == Gainratio_normalized)
				m_ITPCTpriorityCriteria = newPriority;
			else 
				throw new IllegalArgumentException("Wrong selection type, value should be: "
						+ "between 0 and 5");
		}
	}
	/**
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String PCTBconsolidationPercentTipText() {
		
		return "Consolidation percent or number for use after the consolidation process";
	}
	
	/**
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String ITPCTconsolidationPercentHowToSetTipText() {
		return "Way to set the consolidation number to be generated:\n" +
				" * using a fixed value which directly indicates the number nodes to consolidate\n" +
				" * based on a value as a percentage (by default)\n";
	}
	
	/**
	 * Get the value of ITPCTconsolidationPercentHowToSet.
	 *
	 * @return Value of ITPCTconsolidationPercentHowToSet.
	 */
	public SelectedTag getITPCTconsolidationPercentHowToSet() {
		return new SelectedTag(m_ITPCTconsolidationPercentHowToSet,
				TAGS_WAYS_TO_SET_CONSOLIDATION_NUMBER);
	}
	

	/**
	 * Set the value of ITPCTconsolidationPercentHowToSet. Values other than
	 * ConsolidationNumber_Percentage, or ConsolidationNumber_Value will be ignored.
	 *
	 * @param newWayToSetNumberSamples the way to set the number of samples to use
	 * @throws Exception if an option is not supported
	 */
	public void setITPCTconsolidationPercentHowToSet(SelectedTag newWayToSetConsolidationNumber) throws Exception {
		if (newWayToSetConsolidationNumber.getTags() == TAGS_WAYS_TO_SET_CONSOLIDATION_NUMBER) 
		{
			int newEvWay = newWayToSetConsolidationNumber.getSelectedTag().getID();

			if (newEvWay == ConsolidationNumber_Percentage || newEvWay == ConsolidationNumber_Value)
				m_ITPCTconsolidationPercentHowToSet = newEvWay;
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
	public String ITPCTunprunedCTTipText() {
		return "whether to prune or not the partial Consolidated Tree (CT).";
	}
	
	/**
	 * Set whether to prune or not the partial Consolidated Tree (CT). 
	 *
	 * @param unprunedCT whether to prune or not the partial Consolidated Tree (CT).
	 */
	public void setITPCTunprunedCT(boolean unprunedCT) {

		m_ITPCTunprunedCT = unprunedCT;
	}

	/**
	 * Get whether to prune or not the partial Consolidated Tree (CT).
	 *
	 * @return whether to prune or not the partial Consolidated Tree (CT).
	 */
	public boolean getITPCTunprunedCT() {

		return m_ITPCTunprunedCT;
	}

	/**
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String ITPCTcollapseCTTipText() {
		return "whether to collapse or not the partial Consolidated Tree (CT).";
	}
	
	/**
	 * Set whether to collapse or not the partial Consolidated Tree (CT). 
	 *
	 * @param unprunedCT whether to collapse or not the partial Consolidated Tree (CT).
	 */
	public void setITPCTcollapseCT(boolean collapseCT) {

		m_ITPCTcollapseCT = collapseCT;
	}

	/**
	 * Get whether to collapse or not the partial Consolidated Tree (CT).
	 *
	 * @return whether to collapse or not the partial Consolidated Tree (CT).
	 */
	public boolean getITPCTcollapseCT() {

		return m_ITPCTcollapseCT;
	}
}