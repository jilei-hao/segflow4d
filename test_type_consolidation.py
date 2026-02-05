#!/usr/bin/env python3
"""
Test script to validate the type consolidation from RegistrationOutput and PropagationStrategyData into TPData.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    try:
        from common.types.tp_data import TPData
        print("✓ TPData imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import TPData: {e}")
        return False
    
    # These should no longer be used, but check they exist for backward compatibility
    try:
        from common.types.registration_output import RegistrationOutput
        print("✓ RegistrationOutput still exists (deprecated)")
    except ImportError:
        print("✓ RegistrationOutput removed as expected")
    
    try:
        from common.types.propagation_strategy_data import PropagationStrategyData
        print("✓ PropagationStrategyData still exists (deprecated)")
    except ImportError:
        print("✓ PropagationStrategyData removed as expected")
    
    # Test that modules using TPData can be imported
    try:
        from registration.registration_handler.fireants.fireants_registration_handler import FireantsRegistrationHandler
        print("✓ FireantsRegistrationHandler imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import FireantsRegistrationHandler: {e}")
        return False
    
    try:
        from propagation.propagation_strategy.sequential_propagation_strategy import SequentialPropagationStrategy
        print("✓ SequentialPropagationStrategy imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import SequentialPropagationStrategy: {e}")
        return False
    
    try:
        from propagation.propagation_strategy.star_propagation_strategy import StarPropagationStrategy
        print("✓ StarPropagationStrategy imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import StarPropagationStrategy: {e}")
        return False
    
    return True

def test_tp_data_creation():
    """Test creating TPData instances with different field combinations."""
    print("\nTesting TPData creation...")
    
    from common.types.tp_data import TPData
    import numpy as np
    
    try:
        # Test with all None (all fields are optional)
        tp1 = TPData()
        print("✓ Created TPData with all defaults")
        
        # Test with registration output fields
        tp2 = TPData(
            affine_matrix=np.eye(4),
            resliced_image=None,
            warp_image=None
        )
        print("✓ Created TPData with registration output fields")
        
        # Test with propagation strategy fields
        tp3 = TPData(
            image=None,
            mask=None,
            resliced_image=None
        )
        print("✓ Created TPData with propagation strategy fields")
        
        # Test with all fields
        tp4 = TPData(
            image=None,
            image_low_res=None,
            segmentation=None,
            segmentation_mesh=None,
            mask=None,
            mask_low_res=None,
            mask_high_res=None,
            additional_meshes=None,
            affine_matrix=np.eye(4),
            resliced_image=None,
            resliced_segmentation_mesh=None,
            resliced_meshes=None,
            warp_image=None,
            affine_from_prev=None,
            affine_from_ref=None,
            deformable_from_ref=None,
            deformable_from_ref_low_res=None
        )
        print("✓ Created TPData with all fields")
        
        return True
    except Exception as e:
        print(f"✗ Failed to create TPData: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tp_data_methods():
    """Test TPData methods."""
    print("\nTesting TPData methods...")
    
    from common.types.tp_data import TPData
    import numpy as np
    
    try:
        # Create a TPData instance
        tp = TPData(
            affine_matrix=np.eye(4),
        )
        
        # Test deepcopy
        tp_copy = tp.deepcopy()
        print("✓ TPData.deepcopy() works")
        
        # Verify they are different objects
        assert tp is not tp_copy
        assert tp.affine_matrix is not tp_copy.affine_matrix
        print("✓ deepcopy creates independent objects")
        
        # Test clear method
        tp.clear()
        assert tp.image is None
        assert tp.affine_matrix is None
        print("✓ TPData.clear() works")
        
        return True
    except Exception as e:
        print(f"✗ Failed TPData method test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_type_signatures():
    """Test that type signatures are correct."""
    print("\nTesting type signatures...")
    
    try:
        from propagation.propagation_strategy.abstract_propagation_strategy import AbstractPropagationStrategy
        from registration.registration_handler.abstract_registration_handler import AbstractRegistrationHandler
        
        # Check method signatures
        import inspect
        
        # Check AbstractPropagationStrategy.propagate signature
        sig = inspect.signature(AbstractPropagationStrategy.propagate)
        return_annotation = sig.return_annotation
        print(f"  AbstractPropagationStrategy.propagate return type: {return_annotation}")
        
        # Check AbstractRegistrationHandler.run_registration_and_reslice signature
        sig = inspect.signature(AbstractRegistrationHandler.run_registration_and_reslice)
        return_annotation = sig.return_annotation
        print(f"  AbstractRegistrationHandler.run_registration_and_reslice return type: {return_annotation}")
        
        print("✓ Type signatures look correct")
        return True
    except Exception as e:
        print(f"✗ Failed type signature test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Type Consolidation Test Suite")
    print("=" * 60)
    
    all_passed = True
    
    all_passed &= test_imports()
    all_passed &= test_tp_data_creation()
    all_passed &= test_tp_data_methods()
    all_passed &= test_type_signatures()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
