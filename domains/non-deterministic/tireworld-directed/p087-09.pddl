(define (problem tireworld-087-09)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l76 l77 l78 l79 l8 l80 l81 l82 l83 l84 l85 l86 l9 - location)
    (:init (road l0 l43) (road l0 l68) (road l1 l21) (road l1 l24) (road l10 l59) (road l10 l65) (road l11 l15) (road l11 l70) (road l12 l50) (road l12 l60) (road l13 l16) (road l14 l66) (road l15 l11) (road l16 l13) (road l16 l15) (road l17 l19) (road l17 l55) (road l18 l39) (road l18 l63) (road l19 l60) (road l2 l34) (road l2 l48) (road l20 l31) (road l20 l76) (road l21 l80) (road l22 l68) (road l23 l46) (road l23 l49) (road l24 l1) (road l25 l32) (road l26 l37) (road l26 l78) (road l27 l33) (road l27 l79) (road l28 l85) (road l29 l69) (road l3 l4) (road l30 l84) (road l31 l78) (road l32 l75) (road l33 l27) (road l34 l2) (road l34 l56) (road l35 l6) (road l36 l33) (road l37 l26) (road l37 l35) (road l38 l25) (road l38 l64) (road l39 l18) (road l39 l73) (road l4 l9) (road l40 l47) (road l41 l22) (road l41 l80) (road l42 l58) (road l42 l65) (road l43 l0) (road l43 l55) (road l44 l73) (road l44 l74) (road l45 l50) (road l45 l57) (road l46 l23) (road l46 l63) (road l47 l40) (road l47 l83) (road l48 l2) (road l49 l76) (road l5 l62) (road l50 l45) (road l51 l5) (road l52 l14) (road l52 l9) (road l53 l59) (road l54 l3) (road l55 l17) (road l55 l43) (road l56 l34) (road l56 l84) (road l57 l45) (road l57 l74) (road l58 l54) (road l59 l10) (road l59 l53) (road l6 l69) (road l60 l12) (road l61 l53) (road l61 l81) (road l62 l13) (road l62 l5) (road l62 l77) (road l63 l46) (road l64 l38) (road l64 l86) (road l65 l10) (road l65 l42) (road l66 l14) (road l66 l72) (road l67 l29) (road l67 l8) (road l68 l0) (road l69 l29) (road l69 l36) (road l7 l40) (road l7 l79) (road l70 l30) (road l71 l72) (road l71 l86) (road l72 l71) (road l72 l82) (road l73 l39) (road l73 l44) (road l74 l44) (road l75 l28) (road l76 l20) (road l77 l32) (road l77 l62) (road l78 l26) (road l78 l31) (road l79 l27) (road l79 l7) (road l8 l67) (road l8 l85) (road l80 l41) (road l81 l61) (road l81 l83) (road l82 l51) (road l82 l72) (road l83 l47) (road l83 l81) (road l84 l56) (road l85 l8) (road l86 l64) (road l86 l71) (road l9 l52) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l26) (spare-in l27) (spare-in l3) (spare-in l30) (spare-in l31) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l36) (spare-in l37) (spare-in l39) (spare-in l4) (spare-in l40) (spare-in l41) (spare-in l42) (spare-in l43) (spare-in l44) (spare-in l45) (spare-in l46) (spare-in l47) (spare-in l49) (spare-in l5) (spare-in l50) (spare-in l51) (spare-in l52) (spare-in l53) (spare-in l54) (spare-in l55) (spare-in l56) (spare-in l57) (spare-in l58) (spare-in l59) (spare-in l6) (spare-in l60) (spare-in l61) (spare-in l62) (spare-in l63) (spare-in l65) (spare-in l66) (spare-in l68) (spare-in l69) (spare-in l7) (spare-in l70) (spare-in l72) (spare-in l73) (spare-in l74) (spare-in l76) (spare-in l78) (spare-in l79) (spare-in l80) (spare-in l81) (spare-in l82) (spare-in l83) (spare-in l84) (spare-in l9) (vehicle-at l24))
    (:goal (vehicle-at l48))
)