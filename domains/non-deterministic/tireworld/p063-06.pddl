(define (problem tireworld-063-06)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l7 l8 l9 - location)
    (:init (road l0 l49) (road l0 l57) (road l1 l54) (road l1 l9) (road l10 l42) (road l10 l58) (road l10 l62) (road l11 l14) (road l11 l4) (road l12 l3) (road l12 l48) (road l13 l40) (road l13 l9) (road l14 l11) (road l14 l41) (road l15 l17) (road l16 l29) (road l16 l55) (road l17 l15) (road l17 l27) (road l18 l22) (road l18 l34) (road l19 l51) (road l19 l59) (road l2 l33) (road l2 l55) (road l20 l43) (road l20 l51) (road l21 l39) (road l21 l4) (road l22 l18) (road l22 l46) (road l23 l35) (road l23 l60) (road l24 l26) (road l24 l6) (road l25 l31) (road l25 l44) (road l26 l24) (road l26 l41) (road l27 l17) (road l27 l34) (road l28 l42) (road l29 l16) (road l29 l5) (road l3 l12) (road l3 l47) (road l30 l43) (road l30 l56) (road l31 l25) (road l31 l33) (road l32 l37) (road l32 l48) (road l32 l50) (road l33 l2) (road l33 l31) (road l33 l59) (road l34 l18) (road l34 l27) (road l35 l23) (road l35 l8) (road l36 l45) (road l36 l62) (road l37 l32) (road l37 l43) (road l38 l7) (road l38 l8) (road l39 l21) (road l39 l5) (road l4 l11) (road l4 l21) (road l40 l13) (road l40 l46) (road l40 l49) (road l41 l14) (road l41 l26) (road l42 l10) (road l42 l28) (road l42 l52) (road l43 l20) (road l43 l30) (road l43 l37) (road l43 l58) (road l44 l25) (road l44 l45) (road l45 l36) (road l45 l44) (road l45 l54) (road l46 l22) (road l46 l40) (road l47 l3) (road l47 l61) (road l48 l12) (road l48 l32) (road l49 l0) (road l49 l40) (road l5 l29) (road l5 l39) (road l50 l32) (road l50 l56) (road l51 l19) (road l51 l20) (road l52 l42) (road l52 l61) (road l53 l6) (road l53 l60) (road l54 l1) (road l54 l45) (road l55 l16) (road l55 l2) (road l56 l30) (road l56 l50) (road l57 l0) (road l57 l7) (road l58 l10) (road l58 l43) (road l59 l19) (road l59 l33) (road l6 l24) (road l6 l53) (road l60 l23) (road l60 l53) (road l61 l47) (road l61 l52) (road l62 l10) (road l62 l36) (road l7 l38) (road l7 l57) (road l8 l35) (road l8 l38) (road l9 l1) (road l9 l13) (spare-in l0) (spare-in l11) (spare-in l12) (spare-in l14) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l26) (spare-in l27) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l32) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l38) (spare-in l39) (spare-in l4) (spare-in l40) (spare-in l41) (spare-in l42) (spare-in l43) (spare-in l44) (spare-in l46) (spare-in l47) (spare-in l48) (spare-in l49) (spare-in l5) (spare-in l50) (spare-in l51) (spare-in l52) (spare-in l53) (spare-in l55) (spare-in l56) (spare-in l57) (spare-in l59) (spare-in l6) (spare-in l60) (spare-in l61) (spare-in l7) (spare-in l8) (spare-in l9) (vehicle-at l28))
    (:goal (vehicle-at l15))
)