(define (problem tireworld-064-03)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l7 l8 l9 - location)
    (:init (road l0 l21) (road l0 l46) (road l1 l21) (road l1 l9) (road l10 l15) (road l11 l19) (road l11 l4) (road l12 l14) (road l12 l55) (road l13 l26) (road l13 l39) (road l14 l12) (road l14 l60) (road l15 l10) (road l15 l22) (road l15 l32) (road l16 l2) (road l16 l43) (road l17 l29) (road l17 l33) (road l18 l19) (road l18 l34) (road l19 l11) (road l19 l18) (road l2 l16) (road l2 l6) (road l20 l52) (road l20 l56) (road l21 l0) (road l21 l1) (road l21 l50) (road l22 l15) (road l22 l57) (road l23 l30) (road l23 l8) (road l24 l35) (road l24 l50) (road l25 l38) (road l25 l57) (road l26 l13) (road l26 l27) (road l27 l26) (road l27 l62) (road l28 l32) (road l28 l8) (road l29 l17) (road l29 l5) (road l29 l54) (road l3 l31) (road l3 l37) (road l30 l23) (road l30 l63) (road l31 l3) (road l31 l54) (road l32 l15) (road l32 l28) (road l33 l17) (road l33 l58) (road l34 l18) (road l34 l7) (road l35 l24) (road l35 l53) (road l36 l55) (road l36 l6) (road l37 l3) (road l37 l49) (road l38 l25) (road l38 l6) (road l39 l13) (road l39 l5) (road l4 l11) (road l4 l6) (road l40 l42) (road l40 l45) (road l41 l47) (road l41 l63) (road l42 l40) (road l42 l59) (road l43 l16) (road l43 l61) (road l44 l48) (road l44 l51) (road l45 l40) (road l45 l46) (road l46 l0) (road l46 l45) (road l46 l58) (road l47 l41) (road l47 l51) (road l48 l44) (road l48 l62) (road l49 l37) (road l49 l7) (road l5 l29) (road l5 l39) (road l50 l21) (road l50 l24) (road l51 l44) (road l51 l47) (road l52 l20) (road l52 l60) (road l53 l35) (road l53 l56) (road l54 l29) (road l54 l31) (road l55 l12) (road l55 l36) (road l56 l20) (road l56 l53) (road l57 l22) (road l57 l25) (road l58 l33) (road l58 l46) (road l59 l42) (road l6 l2) (road l6 l36) (road l6 l38) (road l6 l4) (road l60 l14) (road l60 l52) (road l61 l43) (road l61 l9) (road l62 l27) (road l62 l48) (road l63 l30) (road l63 l41) (road l7 l34) (road l7 l49) (road l8 l23) (road l8 l28) (road l9 l1) (road l9 l61) (spare-in l11) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l17) (spare-in l18) (spare-in l22) (spare-in l23) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l30) (spare-in l31) (spare-in l32) (spare-in l33) (spare-in l36) (spare-in l39) (spare-in l40) (spare-in l41) (spare-in l42) (spare-in l44) (spare-in l45) (spare-in l46) (spare-in l47) (spare-in l48) (spare-in l49) (spare-in l5) (spare-in l50) (spare-in l51) (spare-in l58) (spare-in l62) (spare-in l63) (spare-in l8) (vehicle-at l59))
    (:goal (vehicle-at l10))
)