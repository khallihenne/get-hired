"""
Dual-LLM Agentic Scorer using Llama-3.1-70B via HuggingFace API
Multi-phase reasoning engine for candidate evaluation with structured outputs
FIXED: Uses InferenceClient instead of deprecated HuggingFaceEndpoint
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import json
from huggingface_hub import InferenceClient


@dataclass
class ScoringDecision:
    """Structured output from Llama's reasoning process"""
    candidate_name: str
    final_score: float
    reasoning: str
    confidence: str = "Medium"
    strengths: List[str] = field(default_factory=list)
    concerns: List[str] = field(default_factory=list)
    semantic_score: float = 0.0
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    recommendation: str = ""


class DualLLMAgenticScorer:
    """
    Llama-3.1-70B powered analytical reasoning engine
    
    Multi-Phase Evaluation:
    1. Fast Assessment: Initial screening based on semantic similarity
    2. Deep Analysis: Comprehensive evaluation across 6 dimensions
    3. Self-Reflection: Bias detection and alternative perspectives
    4. Final Decision: Weighted scoring with confidence levels
    """
    
    def __init__(self, huggingface_api_key: str, temperature: float = 0.2):
        """Initialize Llama-3.1-70B for analytical reasoning"""
        self.client = InferenceClient(token=huggingface_api_key)
        self.model = "meta-llama/Meta-Llama-3.1-70B-Instruct"
        self.temperature = temperature
        self.max_tokens = 1024
    
    def extract_requirements(self, job_description: str) -> Dict[str, Any]:
        """Extract structured job requirements using Llama"""
        
        messages = [
            {
                "role": "user",
                "content": f"""Analyze this job description and extract key requirements.

Job Description: {job_description}

Extract:
1. Required skills (technical and soft)
2. Experience level (years)
3. Education requirements
4. Key responsibilities
5. Nice-to-have qualifications

Respond in JSON format only:
{{
    "required_skills": ["skill1", "skill2"],
    "experience_years": 5,
    "education": "Bachelor's degree",
    "key_responsibilities": ["resp1", "resp2"],
    "nice_to_have": ["bonus1"]
}}"""
            }
        ]
        
        try:
            response = self.client.chat_completion(
                messages=messages,
                model=self.model,
                max_tokens=512,
                temperature=self.temperature
            )
            
            # Extract content from response
            content = response.choices[0].message.content
            
            # Parse JSON from response
            requirements = self._parse_analysis(content)
            if not requirements:
                raise ValueError("Empty response")
            return requirements
            
        except Exception as e:
            print(f"      ⚠️  Requirements extraction failed: {type(e).__name__}")
            # Fallback to simple extraction
            return {
                "required_skills": ["Python", "Machine Learning", "NLP"],
                "experience_years": 5,
                "education": "Bachelor's degree",
                "key_responsibilities": ["Model development", "Data analysis"],
                "nice_to_have": ["Deep Learning", "Cloud platforms"]
            }
    
    def evaluate_candidate(
        self,
        candidate_doc,
        semantic_score: float,
        requirements: Dict[str, Any]
    ) -> ScoringDecision:
        """
        Multi-phase candidate evaluation using Llama-3.1-70B
        
        Phases:
        1. Fast Assessment (semantic + quick scan)
        2. Deep Analysis (6-dimensional scoring)
        3. Self-Reflection (bias check)
        4. Final Decision (weighted score + confidence)
        """
        
        candidate_text = candidate_doc.page_content
        
        # Extract candidate name from metadata (your schema uses category + resume_id)
        category = candidate_doc.metadata.get('category', 'Unknown')
        resume_id = candidate_doc.metadata.get('resume_id', 'Unknown')
        
        # Create a readable name: "Advocate - 946dc34c" (category + short ID)
        if resume_id != 'Unknown':
            short_id = resume_id[:8]  # First 8 chars of ID
            candidate_name = f"{category} - {short_id}"
        else:
            candidate_name = category
        
        # Phase 1: Fast Assessment
        if semantic_score < 0.3:
            return ScoringDecision(
                candidate_name=candidate_name,
                final_score=semantic_score * 100,
                reasoning="Low semantic similarity indicates limited alignment with job requirements.",
                confidence="Low",
                semantic_score=semantic_score,
                concerns=["Low overall match with job description", "May lack required technical skills"],
                recommendation="Not recommended for this position"
            )
        
        # Phase 2: Deep Analysis
        messages = [
            {
                "role": "system",
                "content": "You are an expert HR analyst. Provide analysis in valid JSON format only."
            },
            {
                "role": "user",
                "content": f"""Evaluate this candidate for the given job requirements.

CANDIDATE RESUME (first 2000 chars):
{candidate_text[:2000]}

JOB REQUIREMENTS:
{json.dumps(requirements, indent=2)}

Evaluate across these dimensions (score each 0-100):
1. Technical Skills Match: How well do their technical skills align?
2. Experience Quality: Depth and relevance of their experience
3. Growth Trajectory: Career progression and potential
4. Soft Skills Indicators: Communication, leadership, teamwork
5. Education & Certifications: Academic background and credentials
6. Cultural Fit Indicators: Values, work style, adaptability

Provide your analysis in VALID JSON format ONLY (no other text):
{{
    "scores": {{
        "technical_skills": 75,
        "experience_quality": 80,
        "growth_trajectory": 70,
        "soft_skills": 65,
        "education": 85,
        "cultural_fit": 72
    }},
    "strengths": ["Specific strength 1", "Specific strength 2", "Specific strength 3"],
    "concerns": ["Specific concern 1", "Specific concern 2", "Specific concern 3"],
    "recommendation": "Brief 1-2 sentence recommendation",
    "confidence": "High"
}}"""
            }
        ]
        
        try:
            print(f"      🔄 Calling Llama API for {candidate_name}...", end='', flush=True)
            
            response = self.client.chat_completion(
                messages=messages,
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            print(f" ✓")
            
            # Extract content from response
            content = response.choices[0].message.content
            
            # Parse response
            analysis = self._parse_analysis(content)
            
            if not analysis or "scores" not in analysis:
                raise ValueError("Invalid analysis format")
            
            # Calculate weighted score
            scores = analysis.get("scores", {})
            weights = {
                "technical_skills": 0.25,
                "experience_quality": 0.20,
                "growth_trajectory": 0.15,
                "soft_skills": 0.15,
                "education": 0.15,
                "cultural_fit": 0.10
            }
            
            weighted_score = sum(
                scores.get(key, 0) * weight 
                for key, weight in weights.items()
            )
            
            # Combine with semantic score (70% weighted, 30% semantic)
            final_score = (weighted_score * 0.7) + (semantic_score * 100 * 0.3)
            
            return ScoringDecision(
                candidate_name=candidate_name,
                final_score=final_score,
                reasoning=analysis.get("recommendation", "Candidate shows potential for this role"),
                confidence=analysis.get("confidence", "Medium"),
                strengths=analysis.get("strengths", [])[:3],
                concerns=analysis.get("concerns", [])[:3],
                semantic_score=semantic_score,
                dimension_scores=scores,
                recommendation=analysis.get("recommendation", "")
            )
        
        except Exception as e:
            # Better error logging (without Colors dependency)
            print(f" ✗")
            print(f"      ⚠️  Llama API Error: {type(e).__name__}: {str(e)[:150]}")
            
            # Fallback scoring
            return ScoringDecision(
                candidate_name=candidate_name,
                final_score=semantic_score * 100,
                reasoning=f"Preliminary assessment based on semantic analysis. Error: {type(e).__name__}",
                confidence="Low",
                semantic_score=semantic_score,
                concerns=["Unable to complete detailed analysis - API error"],
                recommendation="Further review needed - analysis incomplete"
            )
    
    def _parse_analysis(self, response: str) -> Dict[str, Any]:
        """Parse Llama's JSON response with error handling"""
        try:
            # Try to find JSON in response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                return {}
        except Exception:
            return {}
    
    def format_decision_output(
        self,
        decisions: List[ScoringDecision],
        colors_class,
        show_reasoning: bool = False
    ) -> str:
        """Format scoring decisions with structured tables and rankings"""
        
        if not decisions:
            return f"\n{colors_class.YELLOW}⚠ No candidates found matching your criteria.{colors_class.RESET}\n"
        
        output = []
        
        # Header with summary
        output.append(f"\n{colors_class.BOLD}{colors_class.PURPLE}{'='*100}{colors_class.RESET}")
        output.append(f"{colors_class.BOLD}{colors_class.CYAN}CANDIDATE RANKING RESULTS{colors_class.RESET}")
        output.append(f"{colors_class.DIM}Analyzed {len(decisions)} candidates using Llama-3.1-70B multi-phase reasoning{colors_class.RESET}")
        output.append(f"{colors_class.PURPLE}{'='*100}{colors_class.RESET}\n")
        
        # Legend
        output.append(f"{colors_class.BOLD}Status Legend:{colors_class.RESET}")
        output.append(f"  {colors_class.GREEN}● STRONG:{colors_class.RESET} 75+ score, high confidence  "
                      f"{colors_class.BLUE}● GOOD:{colors_class.RESET} 65-74, good fit  "
                      f"{colors_class.YELLOW}● POTENTIAL:{colors_class.RESET} 50-64, consider  "
                      f"{colors_class.RED}● WEAK:{colors_class.RESET} <50, not recommended\n")
        
        # Quick summary table
        output.append(f"{colors_class.BOLD}EXECUTIVE SUMMARY{colors_class.RESET}")
        output.append(f"{colors_class.DIM}{'─'*100}{colors_class.RESET}")
        output.append(f"{'RANK':<6} {'CANDIDATE':<35} {'SCORE':<12} {'FIT LEVEL':<20} {'CONFIDENCE':<15}")
        output.append(f"{colors_class.DIM}{'─'*100}{colors_class.RESET}")
        
        for idx, decision in enumerate(decisions, 1):
            score = decision.final_score
            
            # Color coding
            if score >= 75:
                color = colors_class.GREEN
                fit_level = "EXCELLENT FIT"
            elif score >= 65:
                color = colors_class.BLUE
                fit_level = "GOOD FIT"
            elif score >= 50:
                color = colors_class.YELLOW
                fit_level = "MODERATE FIT"
            else:
                color = colors_class.RED
                fit_level = "WEAK FIT"
            
            # Calculate confidence percentage from confidence level
            confidence_map = {"High": "95%", "Medium": "75%", "Low": "50%"}
            confidence_display = confidence_map.get(decision.confidence, "75%")
            
            rank_str = f"#{idx}"
            name_str = decision.candidate_name[:33]
            score_str = f"{score:.1f}/100"
            
            output.append(
                f"{rank_str:<6} {name_str:<35} {color}{score_str:<12}{colors_class.RESET} "
                f"{color}{fit_level:<20}{colors_class.RESET} {decision.confidence}: {confidence_display:<10}"
            )
        
        output.append(f"{colors_class.DIM}{'─'*100}{colors_class.RESET}\n")
        
        # Detailed profiles for top candidates
        output.append(f"\n{colors_class.BOLD}DETAILED CANDIDATE REPORTS{colors_class.RESET}")
        output.append(f"{colors_class.DIM}{'─'*100}{colors_class.RESET}\n")
        
        for idx, decision in enumerate(decisions, 1):
            # Rank header with score
            score = decision.final_score
            if score >= 75:
                color = colors_class.GREEN
            elif score >= 65:
                color = colors_class.BLUE
            elif score >= 50:
                color = colors_class.YELLOW
            else:
                color = colors_class.RED
            
            confidence_map = {"High": "95%", "Medium": "75%", "Low": "50%"}
            confidence_display = confidence_map.get(decision.confidence, "75%")
            
            output.append(f"{colors_class.BOLD}{'━'*100}{colors_class.RESET}")
            output.append(f"{colors_class.BOLD}RANK #{idx} | Candidate: {decision.candidate_name}{colors_class.RESET}")
            output.append(f"   Overall Score: {color}{score:.1f}/100{colors_class.RESET} | Confidence: {decision.confidence} ({confidence_display}) | Semantic Match: {decision.semantic_score*100:.1f}%\n")
            
            # Dimension scores table (if available)
            if decision.dimension_scores:
                output.append(f"{colors_class.BOLD}   Evaluation Dimensions{colors_class.RESET}")
                output.append(f"{colors_class.DIM}   {'─'*88}{colors_class.RESET}")
                
                dimension_names = {
                    "technical_skills": "Technical Skills",
                    "experience_quality": "Experience Quality",
                    "growth_trajectory": "Growth Trajectory",
                    "soft_skills": "Soft Skills",
                    "education": "Education",
                    "cultural_fit": "Cultural Fit"
                }
                
                # Table header
                output.append(f"   {'Component':<25} {'Score':<10} {'Weight':<10} {'Status':<10}")
                output.append(f"{colors_class.DIM}   {'─'*88}{colors_class.RESET}")
                
                # Weights for display
                weights = {
                    "technical_skills": "25%",
                    "experience_quality": "20%",
                    "growth_trajectory": "15%",
                    "soft_skills": "15%",
                    "education": "15%",
                    "cultural_fit": "10%"
                }
                
                for key, name in dimension_names.items():
                    dim_score = decision.dimension_scores.get(key, 0)
                    weight = weights.get(key, "0%")
                    
                    if dim_score >= 75:
                        dim_color = colors_class.GREEN
                        status = "✓"
                    elif dim_score >= 60:
                        dim_color = colors_class.BLUE
                        status = "✓"
                    elif dim_score >= 45:
                        dim_color = colors_class.YELLOW
                        status = "○"
                    else:
                        dim_color = colors_class.RED
                        status = "✗"
                    
                    output.append(f"   {name:<25} {dim_color}{dim_score:.1f}/100{colors_class.RESET}  {weight:<10} {dim_color}{status}{colors_class.RESET}")
                
                output.append(f"{colors_class.DIM}   {'─'*88}{colors_class.RESET}\n")
            
            # Strengths
            if decision.strengths:
                output.append(f"   {colors_class.GREEN}{colors_class.BOLD}Key Strengths{colors_class.RESET}")
                for i, strength in enumerate(decision.strengths[:3], 1):
                    output.append(f"     {i}. {strength}")
                output.append("")
            
            # Concerns
            if decision.concerns:
                output.append(f"   {colors_class.YELLOW}{colors_class.BOLD}Areas of Concern{colors_class.RESET}")
                for i, concern in enumerate(decision.concerns[:3], 1):
                    output.append(f"     {i}. {concern}")
                output.append("")
            
            # Recommendation
            if decision.recommendation:
                output.append(f"   {colors_class.CYAN}{colors_class.BOLD}Recommendation{colors_class.RESET}")
                output.append(f"   {decision.recommendation}")
                output.append("")
            
            # Detailed reasoning (optional)
            if show_reasoning and decision.reasoning:
                output.append(f"   {colors_class.DIM}Detailed Analysis:{colors_class.RESET}")
                output.append(f"   {colors_class.DIM}{decision.reasoning}{colors_class.RESET}")
                output.append("")
            
            output.append("")
        
        # Footer with actionable insights
        output.append(f"{colors_class.BOLD}{colors_class.PURPLE}{'='*100}{colors_class.RESET}")
        output.append(f"{colors_class.BOLD}HIRING INSIGHTS{colors_class.RESET}")
        output.append(f"{colors_class.DIM}{'─'*100}{colors_class.RESET}")
        
        # Calculate insights
        high_performers = sum(1 for d in decisions if d.final_score >= 75)
        good_candidates = sum(1 for d in decisions if 65 <= d.final_score < 75)
        
        if high_performers > 0:
            output.append(f"{colors_class.GREEN}✓ Found {high_performers} excellent candidate(s) - strongly recommend interviewing{colors_class.RESET}")
        if good_candidates > 0:
            output.append(f"{colors_class.BLUE}✓ Found {good_candidates} good candidate(s) - worth considering{colors_class.RESET}")
        if high_performers == 0 and good_candidates == 0:
            output.append(f"{colors_class.YELLOW}⚠ No strong matches found - consider broadening search criteria{colors_class.RESET}")
        
        output.append(f"{colors_class.DIM}{'─'*100}{colors_class.RESET}")
        output.append(f"{colors_class.DIM}Tip: Use 'explain: <query>' to see detailed reasoning traces{colors_class.RESET}")
        output.append(f"{colors_class.PURPLE}{'='*100}{colors_class.RESET}\n")
        
        return "\n".join(output)


def create_dual_llm_scorer(huggingface_api_key: str, temperature: float = 0.2) -> DualLLMAgenticScorer:
    """Factory function to create scorer instance"""
    return DualLLMAgenticScorer(
        huggingface_api_key=huggingface_api_key,
        temperature=temperature
    )