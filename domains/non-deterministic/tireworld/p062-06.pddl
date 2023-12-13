(define (problem tireworld-062-06)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l7 l8 l9 - location)
    (:init (road l0 l26) (road l0 l31) (road l1 l27) (road l10 l41) (road l10 l9) (road l11 l15) (road l11 l58) (road l12 l5) (road l12 l60) (road l13 l2) (road l13 l23) (road l13 l54) (road l14 l28) (road l14 l58) (road l15 l11) (road l15 l59) (road l16 l32) (road l16 l57) (road l17 l48) (road l17 l51) (road l18 l20) (road l18 l24) (road l18 l29) (road l19 l30) (road l19 l45) (road l2 l13) (road l2 l24) (road l20 l18) (road l20 l42) (road l21 l48) (road l21 l6) (road l22 l43) (road l22 l55) (road l23 l13) (road l23 l32) (road l23 l49) (road l23 l8) (road l24 l18) (road l24 l2) (road l25 l36) (road l25 l61) (road l26 l0) (road l26 l39) (road l27 l1) (road l27 l5) (road l28 l14) (road l28 l39) (road l29 l18) (road l29 l40) (road l3 l40) (road l3 l8) (road l30 l19) (road l30 l52) (road l30 l55) (road l31 l0) (road l31 l51) (road l32 l16) (road l32 l23) (road l32 l47) (road l32 l50) (road l33 l37) (road l33 l59) (road l34 l36) (road l34 l60) (road l35 l4) (road l35 l41) (road l36 l25) (road l36 l34) (road l37 l33) (road l37 l56) (road l38 l40) (road l38 l54) (road l39 l26) (road l39 l28) (road l4 l35) (road l40 l29) (road l40 l3) (road l40 l38) (road l40 l42) (road l40 l8) (road l41 l10) (road l41 l35) (road l42 l20) (road l42 l40) (road l43 l22) (road l43 l49) (road l44 l50) (road l44 l9) (road l45 l19) (road l45 l56) (road l46 l54) (road l46 l7) (road l47 l32) (road l47 l54) (road l48 l17) (road l48 l21) (road l49 l23) (road l49 l43) (road l49 l61) (road l5 l12) (road l5 l27) (road l50 l32) (road l50 l44) (road l51 l17) (road l51 l31) (road l52 l30) (road l52 l53) (road l53 l52) (road l53 l7) (road l54 l13) (road l54 l38) (road l54 l46) (road l54 l47) (road l55 l22) (road l55 l30) (road l56 l37) (road l56 l45) (road l57 l16) (road l57 l6) (road l58 l11) (road l58 l14) (road l59 l15) (road l59 l33) (road l6 l21) (road l6 l57) (road l60 l12) (road l60 l34) (road l61 l25) (road l61 l49) (road l7 l46) (road l7 l53) (road l8 l23) (road l8 l3) (road l8 l40) (road l9 l10) (road l9 l44) (spare-in l0) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l30) (spare-in l31) (spare-in l32) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l36) (spare-in l37) (spare-in l39) (spare-in l41) (spare-in l42) (spare-in l43) (spare-in l44) (spare-in l45) (spare-in l46) (spare-in l47) (spare-in l48) (spare-in l49) (spare-in l5) (spare-in l50) (spare-in l51) (spare-in l54) (spare-in l55) (spare-in l56) (spare-in l57) (spare-in l58) (spare-in l59) (spare-in l6) (spare-in l60) (spare-in l61) (spare-in l7) (spare-in l9) (vehicle-at l4))
    (:goal (vehicle-at l1))
)