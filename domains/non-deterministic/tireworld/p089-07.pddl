(define (problem tireworld-089-07)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l76 l77 l78 l79 l8 l80 l81 l82 l83 l84 l85 l86 l87 l88 l9 - location)
    (:init (road l0 l79) (road l0 l81) (road l1 l2) (road l1 l69) (road l10 l35) (road l10 l39) (road l11 l54) (road l11 l60) (road l12 l30) (road l12 l9) (road l13 l31) (road l13 l51) (road l14 l78) (road l14 l81) (road l14 l85) (road l15 l29) (road l15 l83) (road l16 l61) (road l16 l8) (road l17 l54) (road l17 l70) (road l18 l65) (road l18 l88) (road l19 l37) (road l19 l73) (road l2 l1) (road l2 l72) (road l20 l32) (road l20 l49) (road l21 l62) (road l21 l87) (road l22 l24) (road l22 l75) (road l23 l51) (road l23 l86) (road l24 l22) (road l24 l53) (road l25 l6) (road l25 l63) (road l26 l5) (road l26 l75) (road l27 l41) (road l27 l76) (road l28 l49) (road l28 l8) (road l29 l15) (road l29 l55) (road l3 l40) (road l3 l43) (road l30 l12) (road l30 l5) (road l31 l13) (road l31 l88) (road l32 l20) (road l32 l37) (road l33 l36) (road l33 l47) (road l34 l78) (road l35 l10) (road l35 l76) (road l36 l33) (road l36 l85) (road l37 l19) (road l37 l32) (road l38 l72) (road l38 l73) (road l39 l10) (road l39 l56) (road l4 l64) (road l4 l77) (road l40 l3) (road l40 l7) (road l41 l27) (road l41 l9) (road l42 l53) (road l42 l61) (road l43 l3) (road l43 l47) (road l44 l48) (road l44 l80) (road l45 l52) (road l45 l7) (road l46 l6) (road l46 l74) (road l47 l33) (road l47 l43) (road l47 l50) (road l48 l44) (road l48 l58) (road l49 l20) (road l49 l28) (road l5 l26) (road l5 l30) (road l50 l47) (road l50 l84) (road l51 l13) (road l51 l23) (road l52 l45) (road l52 l86) (road l53 l24) (road l53 l42) (road l54 l11) (road l54 l17) (road l55 l29) (road l55 l84) (road l56 l39) (road l56 l65) (road l57 l68) (road l57 l87) (road l58 l48) (road l58 l62) (road l59 l60) (road l59 l71) (road l6 l25) (road l6 l46) (road l60 l11) (road l60 l59) (road l61 l16) (road l61 l42) (road l62 l21) (road l62 l58) (road l63 l25) (road l63 l86) (road l64 l4) (road l64 l74) (road l65 l18) (road l65 l56) (road l66 l68) (road l66 l71) (road l67 l83) (road l68 l57) (road l68 l66) (road l69 l1) (road l69 l85) (road l7 l40) (road l7 l45) (road l70 l17) (road l70 l77) (road l71 l59) (road l71 l66) (road l72 l2) (road l72 l38) (road l73 l19) (road l73 l38) (road l74 l46) (road l74 l64) (road l75 l22) (road l75 l26) (road l76 l27) (road l76 l35) (road l77 l4) (road l77 l70) (road l78 l14) (road l78 l34) (road l79 l0) (road l79 l82) (road l8 l16) (road l8 l28) (road l8 l83) (road l80 l44) (road l80 l82) (road l81 l0) (road l81 l14) (road l82 l79) (road l82 l80) (road l83 l15) (road l83 l67) (road l83 l8) (road l84 l50) (road l84 l55) (road l85 l14) (road l85 l36) (road l85 l69) (road l86 l23) (road l86 l52) (road l86 l63) (road l87 l21) (road l87 l57) (road l88 l18) (road l88 l31) (road l9 l12) (road l9 l41) (spare-in l0) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l17) (spare-in l22) (spare-in l25) (spare-in l29) (spare-in l31) (spare-in l33) (spare-in l36) (spare-in l4) (spare-in l41) (spare-in l45) (spare-in l47) (spare-in l50) (spare-in l55) (spare-in l66) (spare-in l70) (spare-in l71) (spare-in l77) (spare-in l78) (spare-in l83) (spare-in l84) (spare-in l85) (vehicle-at l34))
    (:goal (vehicle-at l67))
)