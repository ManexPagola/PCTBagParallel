package weka.classifiers.trees;

import java.util.Collections;
import java.util.Enumeration;
import java.util.Vector;

import weka.classifiers.Sourcable;
import weka.classifiers.trees.j48.C45ModelSelection;
import weka.classifiers.trees.j48.ModelSelection;
import weka.classifiers.trees.j48Consolidated.C45ConsolidatedModelSelection;
import weka.classifiers.trees.j48Consolidated.C45ConsolidatedModelSelectionParallel;
import weka.classifiers.trees.j48ItPartiallyConsolidated.*;
import weka.classifiers.trees.j48PartiallyConsolidated.C45ModelSelectionExtended;
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

public class J48ItPartiallyConsolidatedParallel extends J48ItPartiallyConsolidated 
implements OptionHandler, Drawable, Matchable, Sourcable,
WeightedInstancesHandler, Summarizable, AdditionalMeasureProducer,
TechnicalInformationHandler {

	
	private static final long serialVersionUID = 1194566856361334394L;
	
	protected int m_numExecutionSlots = 1;
	
	public void setNumExecutionSlots(int numSlots) {
		this.m_numExecutionSlots = numSlots;
	}

	public int getNumExecutionSlots() {
		return this.m_numExecutionSlots;
	}
	
	
	
	/**
	 * Generates the classifier serial or concurrently.
	 * (based on buildClassifier() function of J48Consolidated class)
	 *
	 * @see weka.classifiers.trees.J48Consolidated#buildClassifier(weka.core.Instances)
	 */
	public void buildClassifier(Instances instances) throws Exception {
		int numCore;
		boolean m_static = false;
		
		//if (this.m_numExecutionSlots < 0) {
			//System.out.println("ERROR!!!");
			//throw new Exception("Number of execution slots needs to be >= 0!");
		//}
		
		if ((m_numExecutionSlots != 1) && (m_numExecutionSlots != -1)) {
			numCore =
			        (m_numExecutionSlots == 0) ? Runtime.getRuntime().availableProcessors()
			          : (m_numExecutionSlots < 0) ? -m_numExecutionSlots : m_numExecutionSlots;
			
			if (m_numExecutionSlots < 0) m_static = true;
			
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
			modSelection = new C45ConsolidatedModelSelectionParallel(m_minNumObj, instances, 
					m_useMDLcorrection, m_doNotMakeSplitPointActualValue);
			/** Set the model selection method to force the consolidated decision in each base tree*/
			C45ModelSelectionExtended baseModelToForceDecision = new C45ModelSelectionExtended(m_minNumObj, instances, 
					m_useMDLcorrection, m_doNotMakeSplitPointActualValue);
			// TODO Implement the option reducedErrorPruning of J48
			C45ItPartiallyConsolidatedPruneableClassifierTreeParallel localClassifier =
					new C45ItPartiallyConsolidatedPruneableClassifierTreeParallel(modSelection, baseModelToForceDecision,
							!m_unpruned, m_CF, m_subtreeRaising, !m_noCleanup, m_collapseTree, samplesVector.length,
							m_PCTBpruneBaseTreesWithoutPreservingConsolidatedStructure,
							m_ITPCTpriorityCriteria, !m_ITPCTunprunedCT, m_ITPCTcollapseCT);

			localClassifier.buildClassifierParallel(instances, samplesVector, m_PCTBconsolidationPercent, m_ITPCTconsolidationPercentHowToSet, numCore, m_static);

			m_root = localClassifier;
			m_Classifiers = localClassifier.getSampleTreeVector();
			// // We could get any base tree of the vector as root and use it in the graphical interface
			// // (for example, to visualize it)
			// //m_root = localClassifier.getSampleTreeIth(0);
			
			((C45ModelSelection) modSelection).cleanup();
			((C45ModelSelection) baseModelToForceDecision).cleanup();
		} else {
			super.buildClassifier(instances);
		}
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
	
	public String toString() {
		String st = super.toString();
		
		st += "\n";
		st += "The number of execution slots (threads) to use for constructing the ensemble: ";
		st += Integer.toString(m_numExecutionSlots);
		
		return st;
	}
	
	
	
	
	

}
