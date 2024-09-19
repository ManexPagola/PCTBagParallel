package weka.classifiers.trees;

import java.util.Collections;
import java.util.Enumeration;
import java.util.Vector;

import weka.classifiers.Sourcable;
import weka.classifiers.trees.j48.C45ModelSelection;
import weka.classifiers.trees.j48.ModelSelection;
import weka.classifiers.trees.j48Consolidated.C45ConsolidatedModelSelection;
import weka.classifiers.trees.j48PartiallyConsolidated.C45ModelSelectionExtended;
import weka.classifiers.trees.j48PartiallyConsolidated.C45PartiallyConsolidatedPruneableClassifierTree;
import weka.core.AdditionalMeasureProducer;
import weka.core.Drawable;
import weka.core.Instances;
import weka.core.Matchable;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Summarizable;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;

public class J48PartiallyConsolidatedParallel 
	extends J48PartiallyConsolidated 
	implements OptionHandler, Drawable, Matchable, Sourcable,
	WeightedInstancesHandler, Summarizable, AdditionalMeasureProducer,
	TechnicalInformationHandler {

	
	/**
	 * 
	 */
	private static final long serialVersionUID = -1119606846120036468L;
	protected int m_numExecutionSlots = 1;
	
	public void setNumExecutionSlots(int numSlots) {
		this.m_numExecutionSlots = numSlots;
	}

	public int getNumExecutionSlots() {
		return this.m_numExecutionSlots;
	}

	public String numExecutionSlotsTipText() {
		return "The number of execution slots (threads) to use for constructing the ensemble.";
	}
	
	public Enumeration<Option> listOptions() {
		Vector<Option> newVector = new Vector(2);
		newVector.addElement(new Option(
				"\tNumber of execution slots.\n\t(default 1 - i.e. no parallelism)\n\t(use 0 to auto-detect number of cores)",
				"num-slots", 1, "-num-slots <num>"));
		newVector.addAll(Collections.list(super.listOptions()));
		return newVector.elements();
	}
	
	public void setOptions(String[] options) throws Exception {
		String iterations = Utils.getOption("num-slots", options);
		if (iterations.length() != 0) {
			this.setNumExecutionSlots(Integer.parseInt(iterations));
		} else {
			this.setNumExecutionSlots(1);
		}

		super.setOptions(options);
	}
	
	public String[] getOptions() {
		String[] superOptions = super.getOptions();
		String[] options = new String[superOptions.length + 2];
		int current = 0;
		options[current++] = "-num-slots";
		options[current++] = "" + this.getNumExecutionSlots();
		System.arraycopy(superOptions, 0, options, current, superOptions.length);
		return options;
	}
	
	/**
	 * Klasifikatzailea eraikitzen du seriean / Builds classifier serially 
	 * (J48PartiallyConsolidated klase gurasoko buildClassifiers() metodoa deitzen du)
	 * 
	 * @throws Exception if number of execution slots is lower than 0
	 */
	public void buildClassifier(Instances instances) throws Exception {
		super.buildClassifier(instances);
		if (this.m_numExecutionSlots < 0) {
			throw new Exception("Number of execution slots needs to be >= 0!");
		}
	}
	
	/**
	 * Klasifikatzailea eraikitzen du paraleloan / Builds classifier concurrently
	 * 
	 * @param instances the data to train the classifier with 
	 * @throws Exception if classifier can't be built successfully
	 */
	public void buildClassifiers(Instances instances) throws Exception {
		int numCore;
		
		if (this.m_numExecutionSlots < 0) {
			throw new Exception("Number of execution slots needs to be >= 0!");
		}
		
		if (this.m_numExecutionSlots != 1) {
			numCore = this.m_numExecutionSlots == 0 ? Runtime.getRuntime().availableProcessors() : this.m_numExecutionSlots;
			
			// can classifier tree handle the data?
			getCapabilities().testWithFail(instances);
			
			// remove instances with missing class before generate samples
			instances = new Instances(instances);
			instances.deleteWithMissingClass();
			
			//Generate as many samples as the number of samples with the given instances
			Instances[] samplesVector = generateSamplesParallel(instances, numCore);
			System.out.println("\n " + samplesVector.length + " \n" );
		    //if (m_Debug) printSamplesVector(samplesVector);
			//PRINTZIPIOZ, HONAINO DAGO PARALELIZATURIK
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
		} else {
			this.buildClassifier(instances);
		}
		
		
		
		
	}
	
	
	
	
	
	
	
}
