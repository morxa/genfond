(define (problem tireworld-082-10)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l76 l77 l78 l79 l8 l80 l81 l9 - location)
    (:init (road l0 l45) (road l0 l5) (road l1 l11) (road l1 l16) (road l10 l45) (road l10 l49) (road l11 l1) (road l11 l38) (road l12 l16) (road l12 l51) (road l13 l49) (road l13 l52) (road l13 l55) (road l14 l62) (road l14 l69) (road l15 l41) (road l15 l54) (road l16 l1) (road l16 l12) (road l17 l28) (road l17 l43) (road l18 l37) (road l18 l54) (road l18 l67) (road l19 l55) (road l19 l74) (road l2 l3) (road l2 l50) (road l20 l64) (road l20 l71) (road l21 l26) (road l21 l73) (road l22 l29) (road l22 l52) (road l23 l49) (road l23 l65) (road l24 l4) (road l24 l7) (road l25 l78) (road l25 l81) (road l26 l21) (road l26 l77) (road l27 l46) (road l27 l69) (road l27 l75) (road l28 l17) (road l28 l66) (road l29 l22) (road l29 l64) (road l3 l2) (road l3 l60) (road l30 l35) (road l30 l41) (road l31 l47) (road l31 l80) (road l32 l44) (road l32 l9) (road l33 l63) (road l33 l70) (road l34 l58) (road l34 l63) (road l35 l30) (road l35 l42) (road l36 l38) (road l36 l65) (road l37 l18) (road l37 l44) (road l38 l11) (road l38 l36) (road l39 l53) (road l39 l56) (road l4 l24) (road l4 l51) (road l40 l49) (road l40 l80) (road l41 l15) (road l41 l30) (road l42 l35) (road l42 l72) (road l43 l17) (road l43 l48) (road l44 l32) (road l44 l37) (road l45 l0) (road l45 l10) (road l46 l27) (road l47 l31) (road l47 l56) (road l48 l43) (road l48 l68) (road l49 l10) (road l49 l13) (road l49 l23) (road l49 l40) (road l5 l0) (road l5 l75) (road l50 l2) (road l50 l59) (road l51 l12) (road l51 l4) (road l52 l13) (road l52 l22) (road l53 l39) (road l53 l72) (road l54 l15) (road l54 l18) (road l55 l13) (road l55 l19) (road l56 l39) (road l56 l47) (road l56 l8) (road l57 l6) (road l57 l60) (road l58 l34) (road l58 l81) (road l59 l50) (road l59 l79) (road l6 l57) (road l6 l7) (road l60 l3) (road l60 l57) (road l61 l62) (road l61 l78) (road l62 l14) (road l62 l61) (road l63 l33) (road l63 l34) (road l64 l20) (road l64 l29) (road l65 l23) (road l65 l36) (road l66 l28) (road l66 l70) (road l67 l18) (road l67 l73) (road l68 l48) (road l68 l8) (road l69 l14) (road l69 l27) (road l7 l24) (road l7 l6) (road l70 l33) (road l70 l66) (road l71 l20) (road l71 l79) (road l72 l42) (road l72 l53) (road l73 l21) (road l73 l67) (road l74 l19) (road l74 l76) (road l75 l27) (road l75 l5) (road l75 l77) (road l76 l74) (road l76 l9) (road l77 l26) (road l77 l75) (road l78 l25) (road l78 l61) (road l79 l59) (road l79 l71) (road l8 l56) (road l8 l68) (road l80 l31) (road l80 l40) (road l81 l25) (road l81 l58) (road l9 l32) (road l9 l76) (spare-in l0) (spare-in l13) (spare-in l15) (spare-in l16) (spare-in l2) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l27) (spare-in l34) (spare-in l40) (spare-in l43) (spare-in l45) (spare-in l5) (spare-in l50) (spare-in l51) (spare-in l52) (spare-in l54) (spare-in l56) (spare-in l6) (spare-in l61) (spare-in l65) (spare-in l67) (spare-in l75) (spare-in l76) (spare-in l77) (spare-in l9) (vehicle-at l46))
    (:goal (vehicle-at l10))
)