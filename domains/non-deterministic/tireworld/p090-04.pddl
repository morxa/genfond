(define (problem tireworld-090-04)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l76 l77 l78 l79 l8 l80 l81 l82 l83 l84 l85 l86 l87 l88 l89 l9 - location)
    (:init (road l0 l16) (road l0 l28) (road l0 l63) (road l1 l45) (road l1 l62) (road l10 l28) (road l10 l31) (road l11 l20) (road l12 l4) (road l12 l9) (road l13 l24) (road l13 l77) (road l14 l53) (road l14 l86) (road l15 l53) (road l15 l70) (road l16 l0) (road l16 l88) (road l17 l56) (road l17 l89) (road l18 l26) (road l18 l65) (road l19 l48) (road l19 l50) (road l2 l84) (road l2 l87) (road l20 l11) (road l20 l35) (road l21 l38) (road l21 l47) (road l22 l30) (road l22 l72) (road l23 l37) (road l23 l7) (road l24 l13) (road l24 l39) (road l25 l52) (road l25 l61) (road l26 l18) (road l26 l49) (road l27 l37) (road l27 l60) (road l28 l0) (road l28 l10) (road l28 l59) (road l29 l71) (road l29 l72) (road l3 l5) (road l3 l80) (road l30 l22) (road l30 l68) (road l31 l10) (road l31 l79) (road l32 l52) (road l32 l83) (road l33 l35) (road l33 l38) (road l34 l44) (road l34 l86) (road l35 l20) (road l35 l33) (road l36 l73) (road l36 l89) (road l37 l23) (road l37 l27) (road l38 l21) (road l38 l33) (road l39 l24) (road l39 l43) (road l39 l78) (road l4 l12) (road l4 l76) (road l40 l82) (road l40 l85) (road l41 l42) (road l41 l7) (road l42 l41) (road l42 l51) (road l42 l73) (road l43 l39) (road l43 l56) (road l44 l34) (road l44 l9) (road l45 l1) (road l45 l63) (road l46 l61) (road l46 l87) (road l47 l21) (road l47 l5) (road l47 l86) (road l48 l19) (road l48 l69) (road l49 l26) (road l49 l6) (road l5 l3) (road l5 l47) (road l5 l54) (road l50 l19) (road l50 l55) (road l51 l42) (road l51 l58) (road l52 l25) (road l52 l32) (road l53 l14) (road l53 l15) (road l54 l5) (road l54 l64) (road l55 l50) (road l55 l64) (road l56 l17) (road l56 l43) (road l57 l75) (road l57 l84) (road l58 l51) (road l58 l64) (road l59 l28) (road l59 l80) (road l6 l49) (road l6 l75) (road l60 l27) (road l60 l71) (road l61 l25) (road l61 l46) (road l62 l1) (road l62 l8) (road l63 l0) (road l63 l45) (road l64 l54) (road l64 l55) (road l64 l58) (road l64 l65) (road l65 l18) (road l65 l64) (road l66 l68) (road l66 l81) (road l67 l69) (road l67 l74) (road l68 l30) (road l68 l66) (road l69 l48) (road l69 l67) (road l7 l23) (road l7 l41) (road l7 l8) (road l70 l15) (road l70 l79) (road l71 l29) (road l71 l60) (road l72 l22) (road l72 l29) (road l73 l36) (road l73 l42) (road l74 l67) (road l74 l88) (road l75 l57) (road l75 l6) (road l76 l4) (road l76 l82) (road l77 l13) (road l77 l83) (road l78 l39) (road l78 l81) (road l79 l31) (road l79 l70) (road l79 l85) (road l8 l62) (road l8 l7) (road l80 l3) (road l80 l59) (road l81 l66) (road l81 l78) (road l82 l40) (road l82 l76) (road l83 l32) (road l83 l77) (road l83 l85) (road l84 l2) (road l84 l57) (road l85 l40) (road l85 l79) (road l85 l83) (road l86 l14) (road l86 l34) (road l86 l47) (road l87 l2) (road l87 l46) (road l88 l16) (road l88 l74) (road l89 l17) (road l89 l36) (road l9 l12) (road l9 l44) (spare-in l12) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l23) (spare-in l25) (spare-in l26) (spare-in l32) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l38) (spare-in l4) (spare-in l40) (spare-in l42) (spare-in l44) (spare-in l46) (spare-in l47) (spare-in l48) (spare-in l49) (spare-in l50) (spare-in l52) (spare-in l55) (spare-in l57) (spare-in l6) (spare-in l61) (spare-in l64) (spare-in l65) (spare-in l67) (spare-in l69) (spare-in l7) (spare-in l75) (spare-in l76) (spare-in l82) (spare-in l83) (spare-in l84) (spare-in l85) (spare-in l86) (spare-in l87) (spare-in l9) (vehicle-at l11))
    (:goal (vehicle-at l74))
)