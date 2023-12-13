(define (problem tireworld-088-09)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l76 l77 l78 l79 l8 l80 l81 l82 l83 l84 l85 l86 l87 l9 - location)
    (:init (road l0 l14) (road l0 l53) (road l0 l77) (road l1 l25) (road l1 l56) (road l10 l48) (road l10 l57) (road l11 l24) (road l11 l52) (road l12 l19) (road l12 l5) (road l13 l38) (road l13 l44) (road l14 l0) (road l14 l32) (road l15 l26) (road l15 l51) (road l15 l81) (road l16 l21) (road l16 l63) (road l17 l35) (road l17 l37) (road l18 l27) (road l18 l71) (road l19 l12) (road l19 l7) (road l2 l57) (road l2 l66) (road l2 l71) (road l20 l4) (road l20 l50) (road l21 l16) (road l21 l23) (road l22 l49) (road l22 l6) (road l23 l21) (road l23 l49) (road l24 l11) (road l24 l56) (road l25 l1) (road l25 l55) (road l26 l15) (road l26 l69) (road l27 l18) (road l27 l58) (road l28 l38) (road l28 l83) (road l29 l59) (road l29 l78) (road l3 l32) (road l3 l9) (road l30 l45) (road l30 l8) (road l31 l53) (road l31 l60) (road l32 l14) (road l32 l3) (road l33 l73) (road l33 l85) (road l34 l74) (road l34 l9) (road l35 l17) (road l35 l49) (road l36 l60) (road l36 l75) (road l37 l17) (road l37 l83) (road l38 l13) (road l38 l28) (road l39 l58) (road l39 l59) (road l4 l20) (road l4 l77) (road l40 l64) (road l40 l84) (road l41 l48) (road l41 l75) (road l42 l70) (road l42 l87) (road l43 l85) (road l43 l87) (road l44 l13) (road l44 l81) (road l45 l30) (road l45 l47) (road l46 l80) (road l46 l82) (road l47 l45) (road l47 l68) (road l48 l10) (road l48 l41) (road l49 l22) (road l49 l23) (road l49 l35) (road l5 l12) (road l5 l84) (road l50 l20) (road l50 l78) (road l51 l15) (road l51 l7) (road l52 l11) (road l52 l69) (road l53 l0) (road l53 l31) (road l54 l6) (road l54 l76) (road l55 l25) (road l55 l67) (road l55 l72) (road l56 l1) (road l56 l24) (road l57 l10) (road l57 l2) (road l58 l27) (road l58 l39) (road l59 l29) (road l59 l39) (road l6 l22) (road l6 l54) (road l60 l31) (road l60 l36) (road l60 l65) (road l60 l73) (road l61 l67) (road l61 l86) (road l62 l74) (road l62 l79) (road l63 l16) (road l63 l79) (road l64 l40) (road l64 l72) (road l65 l60) (road l66 l2) (road l66 l68) (road l67 l55) (road l67 l61) (road l68 l47) (road l68 l66) (road l69 l26) (road l69 l52) (road l7 l19) (road l7 l51) (road l70 l42) (road l70 l82) (road l71 l18) (road l71 l2) (road l72 l55) (road l72 l64) (road l73 l33) (road l73 l60) (road l74 l34) (road l74 l62) (road l75 l36) (road l75 l41) (road l76 l54) (road l76 l80) (road l77 l0) (road l77 l4) (road l78 l29) (road l78 l50) (road l79 l62) (road l79 l63) (road l8 l30) (road l80 l46) (road l80 l76) (road l81 l15) (road l81 l44) (road l82 l46) (road l82 l70) (road l83 l28) (road l83 l37) (road l84 l40) (road l84 l5) (road l85 l33) (road l85 l43) (road l85 l86) (road l86 l61) (road l86 l85) (road l87 l42) (road l87 l43) (road l9 l3) (road l9 l34) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l19) (spare-in l2) (spare-in l25) (spare-in l28) (spare-in l29) (spare-in l30) (spare-in l31) (spare-in l32) (spare-in l33) (spare-in l36) (spare-in l37) (spare-in l39) (spare-in l40) (spare-in l41) (spare-in l42) (spare-in l43) (spare-in l44) (spare-in l45) (spare-in l46) (spare-in l47) (spare-in l48) (spare-in l49) (spare-in l51) (spare-in l52) (spare-in l53) (spare-in l54) (spare-in l55) (spare-in l57) (spare-in l59) (spare-in l6) (spare-in l60) (spare-in l61) (spare-in l63) (spare-in l66) (spare-in l68) (spare-in l7) (spare-in l72) (spare-in l75) (spare-in l78) (spare-in l80) (spare-in l81) (spare-in l82) (spare-in l83) (spare-in l86) (vehicle-at l65))
    (:goal (vehicle-at l8))
)